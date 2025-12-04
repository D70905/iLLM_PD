function combined_params = combineParams(pavement_params, subgrade_params)
% Combine pavement parameters and subgrade parameters - Compatible with new interface format
% Input: pavement_params - Pavement structure parameters (first 3 layers)
%        subgrade_params - Subgrade parameters (4th layer and beyond, can be struct or array)
% Output: combined_params - Complete combined parameters

fprintf('🔧 Fixed version parameter combination...\n');

try
    % === [Fix 1] Enhanced input parameter validation ===
    if ~isstruct(pavement_params)
        error('Pavement parameters must be a struct');
    end
    
    % [Fix 2] Flexible handling of different subgrade_params formats
    if ~isstruct(subgrade_params) && ~isempty(subgrade_params)
        % If subgrade_params is not a struct, try to convert
        fprintf('  ⚠️ Subgrade parameters not in struct format, attempting conversion...\n');
        subgrade_params = convertToStructFormat(subgrade_params);
    end
    
    % Initialize combined parameters
    combined_params = pavement_params; % Start with pavement parameters
    
    % === [Fix 3] Validate and process pavement parameters ===
    fprintf('  Validating pavement parameters...\n');
    
    if ~isfield(pavement_params, 'thickness') || isempty(pavement_params.thickness)
        error('Pavement parameters missing thickness field');
    end
    
    if ~isfield(pavement_params, 'modulus') || isempty(pavement_params.modulus)
        error('Pavement parameters missing modulus field');
    end
    
    if ~isfield(pavement_params, 'poisson') || isempty(pavement_params.poisson)
        error('Pavement parameters missing poisson field');
    end
    
    % Ensure pavement parameters are column vectors
    pavement_thickness = pavement_params.thickness(:);
    pavement_modulus = pavement_params.modulus(:);
    pavement_poisson = pavement_params.poisson(:);
    
    fprintf('  Pavement parameters: %d layer thickness, %d layer modulus, %d layer Poisson ratio\n', ...
        length(pavement_thickness), length(pavement_modulus), length(pavement_poisson));
    
    % === [Fix 4] Compatible with multiple subgrade parameter formats ===
    has_subgrade = false;
    subgrade_thickness = [];
    subgrade_modulus = [];
    subgrade_poisson = [];
    subgrade_layers_count = 0;
    
    if isstruct(subgrade_params) && ~isempty(subgrade_params)
        % Format 1: Standard struct format {thickness, modulus, poisson, num_layers}
        if isfield(subgrade_params, 'num_layers') && subgrade_params.num_layers > 0
            fprintf('  Detected standard subgrade struct format\n');
            
            if isfield(subgrade_params, 'thickness') && ~isempty(subgrade_params.thickness)
                subgrade_thickness = subgrade_params.thickness(:);
                has_subgrade = true;
                subgrade_layers_count = length(subgrade_thickness);
            end
            
            if isfield(subgrade_params, 'modulus') && ~isempty(subgrade_params.modulus)
                subgrade_modulus = subgrade_params.modulus(:);
            end
            
            if isfield(subgrade_params, 'poisson') && ~isempty(subgrade_params.poisson)
                subgrade_poisson = subgrade_params.poisson(:);
            end
            
        % Format 2: Direct array struct {thickness: [], modulus: [], poisson: []}
        elseif isfield(subgrade_params, 'thickness') && ~isempty(subgrade_params.thickness)
            fprintf('  Detected direct array subgrade format\n');
            
            subgrade_thickness = subgrade_params.thickness(:);
            has_subgrade = true;
            subgrade_layers_count = length(subgrade_thickness);
            
            if isfield(subgrade_params, 'modulus')
                subgrade_modulus = subgrade_params.modulus(:);
            end
            
            if isfield(subgrade_params, 'poisson')
                subgrade_poisson = subgrade_params.poisson(:);
            end
        end
        
        % Format 3: RoadStructurePPO protected_subgrade_params format
        if isempty(subgrade_thickness) && isfield(subgrade_params, 'protected_subgrade_params')
            fprintf('  Detected PPO protected subgrade format\n');
            protected = subgrade_params.protected_subgrade_params;
            
            if isstruct(protected) && isfield(protected, 'thickness')
                subgrade_thickness = protected.thickness(:);
                has_subgrade = true;
                subgrade_layers_count = length(subgrade_thickness);
                
                if isfield(protected, 'modulus')
                    subgrade_modulus = protected.modulus(:);
                end
                
                if isfield(protected, 'poisson')
                    subgrade_poisson = protected.poisson(:);
                end
            end
        end
    end
    
    % === [Fix 5] Subgrade parameter completion and validation ===
    if has_subgrade && subgrade_layers_count > 0
        fprintf('  Processing %d subgrade layer parameters...\n', subgrade_layers_count);
        
        % Complete missing subgrade parameters
        if isempty(subgrade_modulus) || length(subgrade_modulus) < subgrade_layers_count
            fprintf('    Completing subgrade modulus parameters\n');
            default_subgrade_modulus = [50; 30; 20; 15]; % Decreasing subgrade modulus
            needed = subgrade_layers_count;
            subgrade_modulus = default_subgrade_modulus(1:min(needed, length(default_subgrade_modulus)));
            if needed > length(default_subgrade_modulus)
                % If more layers needed, repeat last value
                extra = repmat(default_subgrade_modulus(end), needed - length(default_subgrade_modulus), 1);
                subgrade_modulus = [subgrade_modulus; extra];
            end
        end
        
        if isempty(subgrade_poisson) || length(subgrade_poisson) < subgrade_layers_count
            fprintf('    Completing subgrade Poisson ratio parameters\n');
            default_subgrade_poisson = [0.45; 0.46; 0.47; 0.48]; % Increasing subgrade Poisson ratio
            needed = subgrade_layers_count;
            subgrade_poisson = default_subgrade_poisson(1:min(needed, length(default_subgrade_poisson)));
            if needed > length(default_subgrade_poisson)
                extra = repmat(default_subgrade_poisson(end), needed - length(default_subgrade_poisson), 1);
                subgrade_poisson = [subgrade_poisson; extra];
            end
        end
        
        % Ensure array length consistency
        min_length = min([length(subgrade_thickness), length(subgrade_modulus), length(subgrade_poisson)]);
        subgrade_thickness = subgrade_thickness(1:min_length);
        subgrade_modulus = subgrade_modulus(1:min_length);
        subgrade_poisson = subgrade_poisson(1:min_length);
        
        % Combine arrays
        combined_params.thickness = [pavement_thickness; subgrade_thickness];
        combined_params.modulus = [pavement_modulus; subgrade_modulus];
        combined_params.poisson = [pavement_poisson; subgrade_poisson];
        
        % [Fix 6] Combine material information (if exists)
        if isfield(pavement_params, 'material') && isfield(subgrade_params, 'material') && ~isempty(subgrade_params.material)
            pavement_materials = pavement_params.material;
            subgrade_materials = subgrade_params.material;
            combined_params.material = [pavement_materials(:); subgrade_materials(:)];
        elseif isfield(pavement_params, 'material')
            % Only pavement materials, add default materials for subgrade
            pavement_materials = pavement_params.material;
            default_subgrade_materials = cell(subgrade_layers_count, 1);
            for i = 1:subgrade_layers_count
                default_subgrade_materials{i} = sprintf('Subgrade Layer %d', i);
            end
            combined_params.material = [pavement_materials(:); default_subgrade_materials];
        end
        
        fprintf('  ✅ Combination complete: Pavement %d layers + Subgrade %d layers = Total %d layers\n', ...
            length(pavement_thickness), length(subgrade_thickness), ...
            length(combined_params.thickness));
        
    else
        % No subgrade layers, use only pavement parameters
        combined_params.thickness = pavement_thickness;
        combined_params.modulus = pavement_modulus;
        combined_params.poisson = pavement_poisson;
        
        fprintf('  ⚠️ No valid subgrade layers, using only pavement structure: %d layers\n', length(pavement_thickness));
    end
    
    % === [Fix 7] Copy all relevant subgrade parameter fields ===
    if isstruct(subgrade_params)
        % Subgrade modeling type
        modeling_fields = {'modeling_type', 'subgrade_modeling', 'boundary_method', 'method'};
        for i = 1:length(modeling_fields)
            field = modeling_fields{i};
            if isfield(subgrade_params, field)
                combined_params.subgrade_modeling = subgrade_params.(field);
                combined_params.boundary_method = subgrade_params.(field);
                break;
            end
        end
        
        % Other subgrade properties
        subgrade_fields = {'subgrade_type', 'subgrade_treatment', 'climate_zone', 'drainage_condition'};
        for i = 1:length(subgrade_fields)
            field = subgrade_fields{i};
            if isfield(subgrade_params, field)
                combined_params.(field) = subgrade_params.(field);
            end
        end
    end
    
    % === [Fix 8] Validate combined result consistency ===
    total_layers = length(combined_params.thickness);
    
    if length(combined_params.modulus) ~= total_layers
        error('Combined modulus array length (%d) inconsistent with thickness array length (%d)', ...
            length(combined_params.modulus), total_layers);
    end
    
    if length(combined_params.poisson) ~= total_layers
        error('Combined Poisson ratio array length (%d) inconsistent with thickness array length (%d)', ...
            length(combined_params.poisson), total_layers);
    end
    
    % === [Fix 9] Ensure all required fields exist ===
    essential_fields = struct(...
        'load_pressure', 0.7, ...
        'load_radius', 21.3, ...
        'traffic_level', 'Heavy Load', ...
        'road_type', 'Expressway', ...
        'vehicle_speed_kmh', 100, ...
        'subgrade_modeling', 'multilayer_subgrade', ...
        'climate_zone', 'Temperate', ...
        'drainage_condition', 'Good');
    
    field_names = fieldnames(essential_fields);
    for i = 1:length(field_names)
        field = field_names{i};
        if ~isfield(combined_params, field)
            combined_params.(field) = essential_fields.(field);
        end
    end
    
    % === [Fix 10] Final validation and summary ===
    fprintf('  Final validation...\n');
    fprintf('    Total layers: %d\n', total_layers);
    fprintf('    Thickness range: [%.1f, %.1f] cm\n', min(combined_params.thickness), max(combined_params.thickness));
    fprintf('    Modulus range: [%.0f, %.0f] MPa\n', min(combined_params.modulus), max(combined_params.modulus));
    fprintf('    Poisson ratio range: [%.3f, %.3f]\n', min(combined_params.poisson), max(combined_params.poisson));
    
    if isfield(combined_params, 'subgrade_modeling')
        fprintf('    Subgrade modeling: %s\n', combined_params.subgrade_modeling);
    end
    
    fprintf('  ✅ Fixed version parameter combination complete\n');
    
catch ME
    fprintf('  ❌ Fixed version parameter combination failed: %s\n', ME.message);
    fprintf('  Error location: %s\n', ME.stack(1).name);
    
    % [Fix 11] Create emergency backup result
    fprintf('  🔄 Creating emergency backup result...\n');
    combined_params = createEmergencyBackupParams(pavement_params);
    
    % Re-throw error for upper level handling
    rethrow(ME);
end
end

function backup_params = createEmergencyBackupParams(pavement_params)
% Create emergency backup parameters

backup_params = pavement_params;

% Ensure basic structure
if ~isfield(backup_params, 'thickness') || isempty(backup_params.thickness)
    backup_params.thickness = [15; 30; 20; 120]; % Default 4 layers
end

if ~isfield(backup_params, 'modulus') || isempty(backup_params.modulus)
    backup_params.modulus = [1200; 600; 200; 50]; % Default modulus
end

if ~isfield(backup_params, 'poisson') || isempty(backup_params.poisson)
    backup_params.poisson = [0.30; 0.25; 0.35; 0.45]; % Default Poisson ratio
end

% Basic fields
backup_params.load_pressure = 0.7;
backup_params.load_radius = 21.3;
backup_params.subgrade_modeling = 'multilayer_subgrade';

fprintf('  Emergency backup parameters: %d layer structure\n', length(backup_params.thickness));
end

function converted_struct = convertToStructFormat(input_data)
% Convert other formats to standard struct format

converted_struct = struct();

try
    if iscell(input_data)
        % Handle cell array format
        if length(input_data) >= 3
            converted_struct.thickness = input_data{1};
            converted_struct.modulus = input_data{2};
            converted_struct.poisson = input_data{3};
            converted_struct.num_layers = length(converted_struct.thickness);
        end
        
    elseif isnumeric(input_data)
        % Handle numeric array format (assume thickness)
        converted_struct.thickness = input_data(:);
        converted_struct.num_layers = length(input_data);
        
        % Add default modulus and Poisson ratio
        default_modulus = repmat(50, converted_struct.num_layers, 1);
        default_poisson = repmat(0.45, converted_struct.num_layers, 1);
        converted_struct.modulus = default_modulus;
        converted_struct.poisson = default_poisson;
        
    else
        % Unrecognized format, create empty structure
        converted_struct.num_layers = 0;
        converted_struct.thickness = [];
        converted_struct.modulus = [];
        converted_struct.poisson = [];
    end
    
catch
    % Conversion failed, return empty structure
    converted_struct.num_layers = 0;
    converted_struct.thickness = [];
    converted_struct.modulus = [];
    converted_struct.poisson = [];
end

fprintf('  Format conversion: Input %s -> Struct (%d layers)\n', class(input_data), converted_struct.num_layers);
end

% [New] Compatibility test function
function testCombineParamsCompatibility()
% Test parameter combination function compatibility

fprintf('=== Testing Fixed Version Parameter Combination Compatibility ===\n');

% Test case 1: Standard format
pavement_params = struct();
pavement_params.thickness = [15; 30; 20];
pavement_params.modulus = [1200; 600; 200];
pavement_params.poisson = [0.30; 0.25; 0.35];

subgrade_params = struct();
subgrade_params.thickness = [120];
subgrade_params.modulus = [50];
subgrade_params.poisson = [0.45];
subgrade_params.num_layers = 1;

try
    result1 = combineParams(pavement_params, subgrade_params);
    fprintf('✅ Test 1 passed: Standard format - %d layers\n', length(result1.thickness));
catch ME1
    fprintf('❌ Test 1 failed: %s\n', ME1.message);
end

% Test case 2: PPO format
subgrade_params2 = struct();
subgrade_params2.protected_subgrade_params = struct();
subgrade_params2.protected_subgrade_params.thickness = [120; 100];
subgrade_params2.protected_subgrade_params.modulus = [50; 30];
subgrade_params2.protected_subgrade_params.poisson = [0.45; 0.47];

try
    result2 = combineParams(pavement_params, subgrade_params2);
    fprintf('✅ Test 2 passed: PPO format - %d layers\n', length(result2.thickness));
catch ME2
    fprintf('❌ Test 2 failed: %s\n', ME2.message);
end

% Test case 3: Empty subgrade
try
    result3 = combineParams(pavement_params, struct());
    fprintf('✅ Test 3 passed: Empty subgrade - %d layers\n', length(result3.thickness));
catch ME3
    fprintf('❌ Test 3 failed: %s\n', ME3.message);
end

fprintf('===============================\n');
end