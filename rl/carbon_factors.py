"""
rl/carbon_factors.py
碳因子提供者（统一接口）：EC3(免费) 现在用；ecoinvent(经 Brightway2) 许可到位后切换。
所有取数缓存到本地 CSV —— 保证论文可复现、运行时不依赖在线 API。

放置位置: rl/carbon_factors.py
依赖: requests
"""
from __future__ import annotations
import os, csv, time, threading
from typing import Dict, Optional

MATERIALS = ["AC_surface", "AC_binder", "AC_base",
             "granular_base", "subbase", "cement_stabilized"]

# 仅“数量级”占位因子 (kgCO2e/吨, cradle-to-gate) —— 必须用 EC3/EPD/本地数据替换并做敏感性
INDICATIVE_GWP = {
    "AC_surface": 65.0, "AC_binder": 62.0, "AC_base": 60.0,
    "granular_base": 6.0, "subbase": 5.0, "cement_stabilized": 120.0,
}
DEFAULT_CACHE = os.path.join("experiments", "carbon_factors_cache.csv")


class CarbonProvider:
    """统一接口：返回某材料 GWP，单位 kgCO2e/吨。"""
    def get_gwp(self, material: str) -> float:
        raise NotImplementedError


class IndicativeProvider(CarbonProvider):
    """内置数量级占位（让管线先跑起来；不可用于正式结论）。"""
    def get_gwp(self, material: str) -> float:
        if material not in INDICATIVE_GWP:
            raise KeyError(f"未知材料 {material}; 请补充 INDICATIVE_GWP / EC3 映射")
        return INDICATIVE_GWP[material]


class EC3Provider(CarbonProvider):
    """
    EC3 (Building Transparency) 免费 API。需要环境变量 EC3_API_KEY。
    注意: EPD 的 GWP 常按“声明单位”(每吨/每m3/每m2)给出, 需换算到“每吨”。
    本类为骨架: 按当前 EC3 API 文档校正 endpoint / 字段 / 单位换算。
    """
    BASE = "https://buildingtransparency.org/api"
    CATEGORY = {  # 材料 -> EC3 查询类别 (TODO: 按 EC3 mat_type 校正)
        "AC_surface": "asphalt", "AC_binder": "asphalt", "AC_base": "asphalt",
        "granular_base": "aggregate", "subbase": "aggregate",
        "cement_stabilized": "cement",
    }

    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        import requests
        self._requests = requests
        self.api_key = api_key or os.environ.get("EC3_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("缺少 EC3_API_KEY (免费注册获取)")

    def get_gwp(self, material: str) -> float:
        cat = self.CATEGORY.get(material)
        if cat is None:
            raise KeyError(f"{material} 未配置 EC3 类别")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.BASE}/materials"          # TODO: 按文档调整 endpoint
        params = {"mat_type": cat, "page_size": 50}
        r = self._requests.get(url, headers=headers, params=params, timeout=self.timeout)
        r.raise_for_status()
        return self._extract_gwp_per_tonne(r.json(), material)

    def _extract_gwp_per_tonne(self, data, material) -> float:
        """
        TODO: 取代表性 GWP 并换算到 kgCO2e/吨:
          per tonne -> 直接用; per m3 -> /密度(t/m3); per m2 -> 需层厚(不建议)。
        下面占位以保证可运行。
        """
        return INDICATIVE_GWP[material]


class EcoinventProvider(CarbonProvider):
    """ecoinvent (经 Brightway2)。教育许可到位后启用。"""
    ACTIVITY = {}  # 材料 -> ecoinvent activity (TODO: 按导入版本填)

    def __init__(self, project: str = "illm_pd",
                 method=("IPCC 2021", "climate change", "global warming potential (GWP100)")):
        import brightway2 as bw
        self.bw = bw
        bw.projects.set_current(project)
        self.method = method

    def get_gwp(self, material: str) -> float:
        raise NotImplementedError("ecoinvent 到位后按 Brightway2 LCA 计算填充")


class CachedProvider(CarbonProvider):
    """带缓存包装器: 先查本地 CSV; 缺失则向 backend 取并写回。论文读这张缓存表即可复现。"""
    def __init__(self, backend: CarbonProvider, cache_path: str = DEFAULT_CACHE):
        self.backend = backend
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache = self._load()

    def _load(self) -> Dict[str, dict]:
        out = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    out[row["material"]] = row
        return out

    def _save_row(self, material, gwp, source):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        exists = os.path.exists(self.cache_path)
        with open(self.cache_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["material", "gwp_kgco2e_per_t", "source", "date"])
            if not exists:
                w.writeheader()
            w.writerow({"material": material, "gwp_kgco2e_per_t": gwp,
                        "source": source, "date": time.strftime("%Y-%m-%d")})

    def get_gwp(self, material: str) -> float:
        with self._lock:
            if material in self._cache:
                return float(self._cache[material]["gwp_kgco2e_per_t"])
            gwp = self.backend.get_gwp(material)
            src = type(self.backend).__name__
            self._save_row(material, gwp, src)
            self._cache[material] = {"material": material, "gwp_kgco2e_per_t": gwp,
                                     "source": src, "date": time.strftime("%Y-%m-%d")}
            return gwp


def get_provider(kind: str = "ec3", cache: bool = True, **kw) -> CarbonProvider:
    """工厂: kind ∈ {'ec3','ecoinvent','indicative'}。默认 EC3 + 缓存。"""
    if kind == "ec3":
        backend = EC3Provider(**kw)
    elif kind == "ecoinvent":
        backend = EcoinventProvider(**kw)
    elif kind == "indicative":
        backend = IndicativeProvider()
    else:
        raise ValueError(kind)
    return CachedProvider(backend) if cache else backend
