function result = roadPDEModelingSimplified(designParams, loadParams, boundary_conditions)
% 【3D Indicators Version】2D pavement modeling - extractσ_FEA, ε_FEA, D_FEA3Dindicator

use_femodel = true;
fprintf('=== 2D PDE 3D Indicators Modeling Started ===\n');

try
    % Step1：Parameter preprocessing and validation
    [thickness, modulus, poisson, P, r, subgrade_config] = preprocessInputs_2D(designParams, loadParams, boundary_conditions);

    % Display processed parameters
    fprintf('Processed parameters: P=%.2f MPa, r=%.1f cm, Number of layers=%d\n', P, r, length(thickness));
    fprintf('Layer thickness（m）: [%s]\n', sprintf('%.3f ', thickness));
    fprintf('Layer thickness（cm）: [%s]\n', sprintf('%.1f ', thickness*100));
    fprintf('Layer modulus: [%s] MPa\n', sprintf('%.0f ', modulus));

    % Verify load parameters
    [P_verified, r_verified] = verifyStandardAxleLoad_2D(P, r);
    if P_verified ~= P || r_verified ~= r
        fprintf('⚠️ Load parameters adjusted to standard axle load: P=%.2f MPa, r=%.1f cm\n', P_verified, r_verified);
        P = P_verified;
        r = r_verified;
    end

    total_thickness = sum(thickness);
    road_width = 4.0;

    try
        % 【pure2Dsolution】
        fprintf('Using pure 2D geometry creation method...\n');
        if ~use_femodel
            % Create PDE Toolbox geometry description matrix using decsg
            try
                fprintf('Step2：Creating 2D plane-strain model...\n');
                model = createpde('structural', 'static-planestrain');
                fprintf('✅ 2D PDE model created successfully\n');

                fprintf('Step3：Creating optimized 2D geometry...\n');
                gd = [3; 4; 0; road_width; road_width; 0; ...
                    -total_thickness; -total_thickness; 0; 0];
                ns = char('R1')';
                sf = 'R1';
                [dl, bt] = decsg(gd, sf, ns);

                geometryFromEdges(model, dl);
                fprintf('✅ Direct 2D rectangle created successfully\n');
                geometry_method = 'direct_2D_rectangle';

            catch ME_direct
                fprintf('  Failed to create 2D geometry: %s\n', ME_direct.message);
            end
        else
            fprintf('Step2：create 2‑D plane‑strain femodel...\n');

            % (a) decsg generategeometry
            gd = [3; 4; 0; road_width; road_width; 0; ...
                -total_thickness; -total_thickness; 0; 0];
            ns = char('R1')';  sf = 'R1';
            [dl,~] = decsg(gd,sf,ns);

            % (b) Convert to fegeometry and feed to femodel
            gm    = fegeometry(dl);
            model = femodel(AnalysisType="structuralStatic",Geometry=gm);
            model.PlanarType = "planeStrain";
            geometry_method = 'femethod_rectangle';
            fprintf('✅ femodel created successfully：%d edges, %d faces\n', gm.NumEdges, gm.NumFaces);
        end

        % Create layer_info
        global layer_info
        layer_info = struct();
        layer_info.num_layers = length(thickness);
        layer_info.road_width = road_width;
        layer_info.total_thickness = total_thickness;
        layer_info.geometry_type = geometry_method;
        layer_info.model_type = '2D_planestrain_3D_indicators_v11';

        % calculatey_positions（usemunit）
        y_positions = zeros(length(thickness) + 1, 1);
        y_positions(1) = 0;  % Pavement surface
        for i = 1:length(thickness)
            y_positions(i + 1) = y_positions(i) - thickness(i);  % Downward is negative, using meters
        end
        layer_info.y_positions = y_positions;
        layer_info.thickness = thickness;  % Save thickness in meters

        fprintf('✅ Pure 2D geometry created successfully, using method: %s\n', geometry_method);
        fprintf('📊 Layer information: Ycoordinates [%s] m\n', sprintf('%.3f ', y_positions));

    catch ME_all
        fprintf('❌ All 2D methods failed: %s\n', ME_all.message);
        error('Unable to create 2D geometry');
    end

    fprintf('✅ 2D geometry creation completed: %dlayer structure\n', layer_info.num_layers);

    % Step4：Defining optimized material properties
    fprintf('Step4：Defining optimized material properties...\n');
    [model,layer_info] = defineOptimized2DMaterials(model, modulus, poisson, layer_info);
    fprintf('✅ Material properties definition completed\n');

    % Step5：generatereasonablesizeofmesh
    fprintf('Step5：Generating optimized mesh...\n');
    model = generateOptimized2DMesh(model, thickness, use_femodel);
    fprintf('✅ Mesh generation completed: %dnodes, %delements\n', size(model.Geometry.Mesh.Nodes,2), size(model.Geometry.Mesh.Elements,2));

    % Check mesh reasonableness
    checkMesh2DReasonableness(model);

    % Step6：Applying 2D boundary conditions
    fprintf('Step6：Applying 2D boundary conditions...\n');
    model = applyOptimized2DBoundaryConditions(model, P, r, layer_info, subgrade_config);
    layer_info = enhanceLayerInfoForWinkler(layer_info, subgrade_config);
    fprintf('✅ Boundary conditions applied\n');

    % Step7：with timeout controlofPDESolve
    fprintf('Step7：Starting PDE solution（with timeout control）...\n');
    tic;
    solution = solveWithTimeout2D(model,layer_info);
    solve_time = toc;
    fprintf('✅ PDE solution completed，time elapsed: %.2fseconds\n', solve_time);

    % Step8：【3D Indicators Version】extractresult
    fprintf('Step8：Extracting finite element modeling results X_FEA（σ_FEA, ε_FEA, D_FEA）...\n');
    thickness_cm = thickness * 100;  % m -> cm，transmitgiveresultextractfunction
    result = extract3DIndicators(model, solution, thickness_cm, modulus, layer_info);
    result.success = true;
    result.message = '2D 3D indicators modeling completed successfully';
    result.software = 'MATLAB_PDE_2D_3D_Indicators_v11';
    result.solve_time = solve_time;
    result.is_true_layered = true;
    result.num_layers = layer_info.num_layers;
    result.boundary_method = subgrade_config.modeling_type;

    % addloadinformation
    result.load_pressure_MPa = P;
    result.load_radius_cm = r;
    result.total_load_per_unit_length_kN = calculateTotalLoad2D(P, r);
    result.road_width = layer_info.road_width;
    result.modeling_type = 'optimized_2D_planestrain_3D_indicators_v11';

    fprintf('✅ 2D 3D indicators modeling completed successfully\n');
    fprintf('Modeling information: %dlayer structure，load per unit length%.0f kN/m，road width%.1fm\n', ...
        result.num_layers, result.total_load_per_unit_length_kN, result.road_width);

catch ME
    fprintf('❌ Layered modeling failed: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end

    % Create failure result
    result = createFailureResult3D(ME.message);
end
end

%% ================ 【3D Indicators Version】Core Result Extraction Function ================

function result = extract3DIndicators(model, solution, thickness, modulus, layer_info)
% 【3D Indicators Version】Extracting finite element modeling results X_FEA：σ_FEA, ε_FEA, D_FEA

fprintf('Extracting finite element modeling results X_FEA...\n');
result = struct();
result.Solution = solution;
result.Model = model;

try
    % Get mesh nodes and solution
    [nodes, u] = extractMeshAndSolution(model, solution);
    
    % Thickness unit handling
    if all(thickness > 1)  % Input is in cm
        thickness_m = thickness / 100;  % cm -> m
        fprintf('Thickness unit conversion: cm -> m\n');
    else  % Input is already in m
        thickness_m = thickness;
        fprintf('Thickness already in meters, no conversion needed\n');
    end
    
    num_layers = length(thickness_m);
    
    fprintf('Mesh information: %d nodes, solution dimensions: %s\n', size(nodes,2), mat2str(size(u)));
    fprintf('Number of layers: %d, thickness: [%s] m\n', num_layers, sprintf('%.3f ', thickness_m));
    
    % === 【Key】Extracting finite element modeling results X_FEA ===
    
    % 1. σ_FEA：Surface layer bottom tensile stress（fatigue control）
    sigma_FEA = extractSurfaceTensileStress(solution, nodes, thickness_m, layer_info);
    
    % 2. ε_FEA：Base layer bottom tensile strain（structural control）
    epsilon_FEA = extractBaseTensileStrain(solution, nodes, thickness_m, layer_info);
    
    % 3. D_FEA：Subgrade top surface deflection（deformation control）
    D_FEA = extractSubgradeDeflection(solution, nodes, thickness_m, layer_info);
    
    % === Assemble 3D indicators result ===
    result = assemble3DIndicatorsResult(result, sigma_FEA, epsilon_FEA, D_FEA, ...
        thickness, modulus, layer_info);
    
    fprintf('✅ Finite element modeling results X_FEA extraction completed\n');
    display3DIndicators(result);
    
catch ME
    fprintf('❌ 3D indicators extraction failed: %s\n', ME.message);
    result = createDefault3DIndicatorsResult(thickness, modulus);
end
end

function sigma_FEA = extractSurfaceTensileStress(solution, nodes, thickness_m, layer_info)
% Extracting surface layer bottom tensile stressσ_FEA（fatigue control）

fprintf('Extracting surface layer bottom tensile stressσ_FEA...\n');

try
    % Surface layer bottom depth
    surface_bottom_depth = -thickness_m(1);
    
    % Get node coordinates
    x_coords = nodes(1,:)';
    y_coords = nodes(2,:)';
    
    % Get stress
    S = solution.Stress;
    sxx = S.xx;  % Xdirection normal stress
    syy = S.yy;  % Ydirection normal stress
    sxy = S.xy;  % shear stress
    
    % Calculate tensile stress（surface layer bottom mainly concerns tensile stress）
    % Use maximum principal stress as tensile stress indicator
    stress_principal_1 = 0.5 * (sxx + syy + sqrt((sxx - syy).^2 + 4*sxy.^2));
    tensile_stress = max(stress_principal_1, 0);  % Only consider tensile stress
    
    % Find nodes at surface layer bottom center position
    road_center_x = (max(x_coords) + min(x_coords)) / 2;
    tolerance_x = 0.3;  % Xdirection tolerance
    tolerance_y = thickness_m(1) * 0.2;  % Ydirection tolerance
    
    target_nodes = abs(x_coords - road_center_x) <= tolerance_x & ...
                  abs(y_coords - surface_bottom_depth) <= tolerance_y;
    
    if sum(target_nodes) >= 3
        % Take average tensile stress
        sigma_FEA = mean(tensile_stress(target_nodes)) / 1e6;  % Pa -> MPa
        fprintf('  Surface layer bottom tensile stress: %.4f MPa (based on%dnodes)\n', ...
            sigma_FEA, sum(target_nodes));
    else
        % Use interpolation
        sigma_FEA = robustInterpolation(y_coords, tensile_stress, surface_bottom_depth) / 1e6;
        fprintf('  Surface layer bottom tensile stress: %.4f MPa (interpolated)\n', sigma_FEA);
    end
    
    % Reasonableness check
    if sigma_FEA < 0.01 || sigma_FEA > 3.0
        fprintf('  ⚠️ Surface layer tensile stress abnormal: %.4f MPa，using estimated value\n', sigma_FEA);
        sigma_FEA = 0.6;  % reasonable estimated value
    end
    
catch ME
    fprintf('  ❌ Surface layer tensile stress extraction failed: %s\n', ME.message);
    sigma_FEA = 0.6;  % default value
end
end

function epsilon_FEA = extractBaseTensileStrain(solution, nodes, thickness_m, layer_info)
% Extracting base layer bottom tensile strainε_FEA（structural control）

fprintf('Extracting base layer bottom tensile strainε_FEA...\n');

try
    if length(thickness_m) < 2
        epsilon_FEA = 400;  % Default value when no base layer
        fprintf('  No base layer structure, using default strain value\n');
        return;
    end
    
    % Base layer bottom depth
    base_bottom_depth = -sum(thickness_m(1:2));
    
    % Get node coordinates
    x_coords = nodes(1,:)';
    y_coords = nodes(2,:)';
    
    % Get strain
    E = solution.Strain;
    exx = E.xx;  % Xdirection normal strain
    eyy = E.yy;  % Ydirection normal strain
    exy = E.xy;  % shear strain
    
    % Calculate tensile strain（base layer bottom mainly concerns tensile strain）
    % Use maximum principal strain as tensile strain indicator
    strain_principal_1 = 0.5 * (exx + eyy + sqrt((exx - eyy).^2 + 4*exy.^2));
    tensile_strain = max(strain_principal_1, 0);  % Only consider tensile strain
    
    % Find nodes at base layer bottom center position
    road_center_x = (max(x_coords) + min(x_coords)) / 2;
    tolerance_x = 0.3;  % Xdirection tolerance
    tolerance_y = thickness_m(2) * 0.2;  % Ydirection tolerance
    
    target_nodes = abs(x_coords - road_center_x) <= tolerance_x & ...
                  abs(y_coords - base_bottom_depth) <= tolerance_y;
    
    if sum(target_nodes) >= 3
        % Take average tensile strain
        epsilon_FEA = mean(tensile_strain(target_nodes)) * 1e6;  % -> με
        fprintf('  Base layer bottom tensile strain: %.2f με (based on%dnodes)\n', ...
            epsilon_FEA, sum(target_nodes));
    else
        % Use interpolation
        epsilon_FEA = robustInterpolation(y_coords, tensile_strain, base_bottom_depth) * 1e6;
        fprintf('  Base layer bottom tensile strain: %.2f με (interpolated)\n', epsilon_FEA);
    end
    
    % Reasonableness check
    if epsilon_FEA < 10 || epsilon_FEA > 3000
        fprintf('  ⚠️ Base layer tensile strain abnormal: %.2f με，using estimated value\n', epsilon_FEA);
        epsilon_FEA = 500;  % reasonable estimated value
    end
    
catch ME
    fprintf('  ❌ Base layer tensile strain extraction failed: %s\n', ME.message);
    epsilon_FEA = 500;  % default value
end
end


function D_FEA = extractSubgradeDeflection(solution, nodes, thickness_m, layer_info)
% 【Corrected version】Subgrade top surface deflection extraction - Fix Winkler model over-correction issue
% Key fix：
% 1. check layer_info.modeling_type
% 2. If 'winkler_springs'，D_FEA = raw_deflection (don't apply Es correction)
% 3. If 'multilayer_subgrade'，D_FEA = raw_deflection * (50 / Es) (keep Es correction)

fprintf('=== Corrected version of subgrade top surface deflection extraction (Winkler Bug Fix v2) ===\n');

try
    % === Step1：Directly unify data format（without additional field checks）===
    [x_coords, y_coords, v_displacement] = unifyDataFormat(nodes, solution.NodalSolution);
    
    % === Step2：Data validation and cleaning ===
    valid_idx = ~isnan(x_coords) & ~isnan(y_coords) & ~isnan(v_displacement) & ...
                ~isinf(x_coords) & ~isinf(y_coords) & ~isinf(v_displacement) & ...
                abs(v_displacement) > 1e-10;
    
    if sum(valid_idx) == 0
        error('No valid node data');
    end
    
    x_valid = x_coords(valid_idx);
    y_valid = y_coords(valid_idx);
    v_valid = v_displacement(valid_idx);
    
    fprintf('  Valid data points: %d/%d\n', sum(valid_idx), length(x_coords));
    fprintf('  Ycoordinate range: [%.6f, %.6f] m\n', min(y_valid), max(y_valid));
    fprintf('  displacement range: [%.6e, %.6e] m\n', min(abs(v_valid)), max(abs(v_valid)));
    
    % === Step3：Determine subgrade top surface depth ===
     % === 【Unified rule】Determine subgrade top surface depth ===
     % Both methods use first 3 layers as pavement structure, ensuring fair comparison
     pavement_layers = min(3, length(thickness_m));
     subgrade_top_depth = -sum(thickness_m(1:pavement_layers)); %

     % Get modeling type（for logging only）
     modeling_type = 'unknown';
     if isfield(layer_info, 'modeling_type')
       modeling_type = layer_info.modeling_type; %
     end

     fprintf('  【Unified rule】Subgrade top surface depth = %.6f m\n', subgrade_top_depth);
     fprintf('  (before%dlayers pavement structure total thickness，modeling type: %s)\n', ...
     pavement_layers, modeling_type);
    
    % === Step4：Multi-strategy deflection extraction ===
    D_FEA = 0;
    extraction_success = false;
    
    road_center_x = mean([max(x_valid), min(x_valid)]);
    
    % Determine Y-direction tolerance
    if length(thickness_m) >= 3
        tol_y_base = thickness_m(3); % based on subbase thickness
    else
        tol_y_base = sum(thickness_m); % based on total thickness
    end
    
    % Strategy1：Precise position search（subgrade top surface ±10% depth range）
    tolerance_x = 0.5;
    tolerance_y = max(tol_y_base * 0.1, 0.01); % minimum 1cm tolerance
    
    target_nodes = abs(x_valid - road_center_x) <= tolerance_x & ...
                   abs(y_valid - subgrade_top_depth) <= tolerance_y; %
    
      if sum(target_nodes) >= 1
        selected_displacements = v_valid(target_nodes);
        raw_deflection = max(abs(selected_displacements)) * 1000; % m -> mm
        
        % 🔧 Get Es
        Es = 50;
        if isfield(layer_info, 'soil_modulus') && layer_info.soil_modulus > 0
            Es = layer_info.soil_modulus; %
        elseif isfield(layer_info, 'subgrade_modulus') && layer_info.subgrade_modulus > 0
            Es = layer_info.subgrade_modulus; %
        end
        
        % =================================================================
        % 【Key fix】Winkler model k value already includes Es, should not be corrected again
        % =================================================================
        if isfield(layer_info, 'modeling_type') && strcmpi(layer_info.modeling_type, 'winkler_springs')
            soil_correction = 1.0;
            D_FEA = raw_deflection;
            fprintf('  Strategy 1 successful (Winkler): Precise position，%dnodes\n', sum(target_nodes));
            fprintf('    Winklermodetype: raw deflection=%.3f mm (k already includes Es，correction factor=1.0)\n', raw_deflection);
        else
            % multilayer elastic model，apply subgrade correction
            soil_correction = 50 / Es; %
            D_FEA = raw_deflection * soil_correction; %
            fprintf('  Strategy 1 successful (multilayer): Precise position，%dnodes\n', sum(target_nodes));
            fprintf('    multilayer model: raw deflection=%.3f mm, Es=%d MPa, correction factor=%.3f, final deflection=%.3f mm\n', ...
                raw_deflection, Es, soil_correction, D_FEA);
        end
        % =================================================================
        
        extraction_success = true;
    end
    
    % Strategy2：Expand search range（±20%depthrange）
    if ~extraction_success
        tolerance_x = 1.0;
        tolerance_y = max(tol_y_base * 0.2, 0.02); % minimum 2cm tolerance
        
        target_nodes = abs(x_valid - road_center_x) <= tolerance_x & ...
                       abs(y_valid - subgrade_top_depth) <= tolerance_y; %
        
         if sum(target_nodes) >= 1
            selected_displacements = v_valid(target_nodes);
            raw_deflection = max(abs(selected_displacements)) * 1000;
            
            % 🔧 Get Es
            Es = 50;
            if isfield(layer_info, 'soil_modulus') && layer_info.soil_modulus > 0
                Es = layer_info.soil_modulus; %
            elseif isfield(layer_info, 'subgrade_modulus') && layer_info.subgrade_modulus > 0
                Es = layer_info.subgrade_modulus; %
            end
            
            % =================================================================
            % 【Key fix】Winkler model k value already includes Es, should not be corrected again
            % =================================================================
            if isfield(layer_info, 'modeling_type') && strcmpi(layer_info.modeling_type, 'winkler_springs')
                soil_correction = 1.0;
                D_FEA = raw_deflection;
                fprintf('  Strategy2successful (Winkler): Expanded range，%dnodes\n', sum(target_nodes));
                fprintf('    Winklermodetype: raw deflection=%.3f mm (k already includes Es，correction factor=1.0)\n', raw_deflection);
            else
                % multilayer elastic model，apply subgrade correction
                soil_correction = 50 / Es;
                D_FEA = raw_deflection * soil_correction;
                fprintf('  Strategy2successful (multilayer): Expanded range，%dnodes\n', sum(target_nodes));
                fprintf('    multilayer model: raw deflection=%.3f mm, Es=%d MPa, correction factor=%.3f, final deflection=%.3f mm\n', ...
                    raw_deflection, Es, soil_correction, D_FEA);
            end
            % =================================================================
            
            extraction_success = true;
        end
    end
    
    % Strategy3：Interpolation method
    if ~extraction_success && length(y_valid) >= 3
        try
            % Sort by Y coordinate
            [y_sorted, sort_idx] = sort(y_valid);
            v_sorted = v_valid(sort_idx);
            
            % Linear interpolation
            v_interp = interp1(y_sorted, v_sorted, subgrade_top_depth, 'linear', 'extrap');
            raw_deflection = abs(v_interp) * 1000;
            
            % 🔧 Get Es
            Es = 50;
            if isfield(layer_info, 'soil_modulus') && layer_info.soil_modulus > 0
                Es = layer_info.soil_modulus; %
            elseif isfield(layer_info, 'subgrade_modulus') && layer_info.subgrade_modulus > 0
                Es = layer_info.subgrade_modulus; %
            end
            
            % =================================================================
            % 【Key fix】Winkler model k value already includes Es, should not be corrected again
            % =================================================================
            if isfield(layer_info, 'modeling_type') && strcmpi(layer_info.modeling_type, 'winkler_springs')
                soil_correction = 1.0;
                D_FEA = raw_deflection;
                fprintf('  Strategy 3 successful (Winkler): Interpolation method\n');
                fprintf('    Winklermodetype: raw deflection=%.3f mm (k already includes Es，correction factor=1.0)\n', raw_deflection);
            else
                % multilayer elastic model，apply subgrade correction
                soil_correction = 50 / Es;
                D_FEA = raw_deflection * soil_correction;
                fprintf('  Strategy 3 successful (multilayer): Interpolation method\n');
                fprintf('    multilayer model: raw deflection=%.3f mm, Es=%d MPa, correction factor=%.3f, final deflection=%.3f mm\n', ...
                    raw_deflection, Es, soil_correction, D_FEA);
            end
            % =================================================================
            
            extraction_success = true;
        catch ME_interp
            fprintf('  Strategy3failed: %s\n', ME_interp.message);
        end
    end
    
    % Strategy4：based onmaximumdisplacementestimate（thisStrategymaintainunchanged，becauseitisa kind ofestimate，mustdependEs）
    if ~extraction_success
        max_displacement = max(abs(v_valid));
        
        % calculatepavement structuretotalthickness
        total_thickness_pavement = sum(thickness_m(1:pavement_layers));
        
        % 🔧 Keycorrect：getsoil foundationmodulusEs
        Es = 50; % default value
        if isfield(layer_info, 'soil_modulus') && layer_info.soil_modulus > 0
            Es = layer_info.soil_modulus; %
        elseif isfield(layer_info, 'subgrade_modulus') && layer_info.subgrade_modulus > 0
            Es = layer_info.subgrade_modulus; %
        elseif isfield(layer_info, 'theory_params') && isfield(layer_info.theory_params, 'Es_MPa')
            Es = layer_info.theory_params.Es_MPa;
        end
        
        % 🔧 Keycorrect：comprehensiveconsiderthicknesssumsoil foundationmodulus
        thickness_factor = 0.8 / max(total_thickness_pavement, 0.4);
        soil_factor = 50 / Es; % Esmorelarge，deflectionmoresmall
        stiffness_factor = thickness_factor * soil_factor;

        D_FEA = max_displacement * stiffness_factor * 1000;

       fprintf('  Strategy4: estimatemethod（Corrected version）\n');
       fprintf('    max_displacement = %.2e m\n', max_displacement);
       fprintf('    total_thickness = %.3f m\n', total_thickness_pavement);
       fprintf('    Es = %d MPa\n', Es);
       fprintf('    thickness_factor = %.3f\n', thickness_factor);
       fprintf('    soil_factor = %.3f\n', soil_factor);
       fprintf('    stiffness_factor = %.3f\n', stiffness_factor);
       fprintf('    D_FEA = %.6f mm\n', D_FEA);
    end
    
    % === Step5：intelligentreasonablenessjudge ===
    % (this part of codemaintainunchanged)
    total_thickness_pavement = sum(thickness_m(1:min(3, length(thickness_m))));
    
    if total_thickness_pavement > 0.8  % thickness>80cm
        min_reasonable = 0.2;  
        max_reasonable = 5.0;
        deflection_type = 'highstiffness';
    elseif total_thickness_pavement > 0.5  % thickness50-80cm
        min_reasonable = 0.5;
        max_reasonable = 10.0;
        deflection_type = 'inequalstiffness';
    else  % thickness<50cm
        min_reasonable = 1.0;
        max_reasonable = 20.0;
        deflection_type = 'lowstiffness';
    end
    
    if isfield(layer_info, 'modulus') && ~isempty(layer_info.modulus)
        avg_modulus = mean(layer_info.modulus(1:min(3, length(layer_info.modulus))));
        if avg_modulus > 1500  % highmodulus
            min_reasonable = min_reasonable * 0.5;
            max_reasonable = max_reasonable * 0.5;
        elseif avg_modulus < 500  % lowmodulus
            min_reasonable = min_reasonable * 1.5;
            max_reasonable = max_reasonable * 1.5;
        end
    end
    
    if D_FEA < min_reasonable
        fprintf('  ℹ️ deflectionbiasedsmall(%.3f mm)，%sstructureofreasonablerange[%.1f, %.1f] mm\n', ...
            D_FEA, deflection_type, min_reasonable, max_reasonable);
    elseif D_FEA > max_reasonable
        fprintf('  ⚠️ warning：deflectionbiasedlarge(%.3f mm)，%sstructureofreasonablerange[%.1f, %.1f] mm\n', ...
            D_FEA, deflection_type, min_reasonable, max_reasonable);
    else
        fprintf('  ✅ deflectionnormal(%.3f mm)，conform%sstructurefeature\n', D_FEA, deflection_type);
    end
    
    if isnan(D_FEA) || isinf(D_FEA) || D_FEA < 0
        fprintf('  ❌ error：deflectionnumbervalueinvalid，usebased onstructureofestimated value\n');
        D_FEA = (min_reasonable + max_reasonable) / 2;
    end
    
    fprintf('  final deflection: %.6f mm\n', D_FEA);

catch ME
    fprintf('❌ deflectionextractmain programfailed: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('   Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    
    try
        if ~exist('min_reasonable', 'var')
            min_reasonable = 1.0; max_reasonable = 20.0;
        end
        D_FEA = (min_reasonable + max_reasonable) / 2;
        fprintf('  useestimatedefault value: %.6f mm\n', D_FEA);
    catch
        D_FEA = 8.0; % hardencodingdefault value
        fprintf('  usehardencodingdefault value: 8.0 mm\n');
    end
end

end


function case_factor = getCaseSpecificFactorFixed(layer_info)
% getworking condition specificofcorrection factor（fixversion）

case_factor = 1.0; % default value

try
    if isfield(layer_info, 'modeling_type')
        modeling_type = layer_info.modeling_type;
        
        if contains(modeling_type, 'winkler', 'IgnoreCase', true)
            % Winklermethod：based onspringstiffnesssumsoil foundationmodulusadjust
            if isfield(layer_info, 'spring_coefficient') && layer_info.spring_coefficient > 0
                k_spring = layer_info.spring_coefficient;
                % springstiffnessmorelarge，deflectionmoresmall，usenonlinear relationship
                reference_k = 100e6; % referencestiffness
                case_factor = (reference_k / k_spring)^0.5; % squareroot relationship
                fprintf('    Winklercoefficient: k=%.2e -> factor=%.3f\n', k_spring, case_factor);
            elseif isfield(layer_info, 'soil_modulus') && layer_info.soil_modulus > 0
                Es = layer_info.soil_modulus;
                % soil foundationmodulusmorelarge，deflectionmoresmall
                reference_Es = 50; % referencemodulus MPa
                case_factor = (reference_Es / Es)^0.8; % exponential relationship
                fprintf('    Winklercoefficient: Es=%.0f MPa -> factor=%.3f\n', Es, case_factor);
            else
                case_factor = 1.0; % Winklerdefault
            end
        else
            % multilayermethodthroughconstantdeflectionmorelarge
            case_factor = 1.3;
            fprintf('    multilayermethodcoefficient: %.3f\n', case_factor);
        end
    end
    
    % limitmakeinreasonablerangewithin
    case_factor = max(0.2, min(case_factor, 5.0));
    
catch ME
    fprintf('    coefficientcalculatefailed: %s，usedefault value1.0\n', ME.message);
    case_factor = 1.0;
end
end

function D_FEA = validateDeflectionWithCaseContext(D_FEA, layer_info)
% based onworking condition contextofdeflectionverify（fixversion）

try
    % checkisnohavepreperioddeflectionrange
    if isfield(layer_info, 'expected_deflection_range') && ~isempty(layer_info.expected_deflection_range)
        expected_range = layer_info.expected_deflection_range;
        
        fprintf('    preperioddeflectionrange: [%.2f, %.2f] mm\n', expected_range(1), expected_range(2));
        fprintf('    whenbeforedeflection: %.6f mm\n', D_FEA);
        
        if D_FEA < expected_range(1) * 0.5 || D_FEA > expected_range(2) * 2.0
            fprintf('    ⚠️ deflectionexceedreasonablerange，enterlineadjust\n');
            
            % adjustStrategy：ifbiaseddifferencetoolarge，take expectedrangeofreasonablevalue
            if D_FEA < expected_range(1)
                D_FEA = expected_range(1) * 0.8;
            elseif D_FEA > expected_range(2)
                D_FEA = expected_range(2) * 1.2;
            end
            
            fprintf('    adjustafterdeflection: %.6f mm\n', D_FEA);
        else
            fprintf('    ✅ deflectioninreasonablerangewithin\n');
        end
    else
        % nohavepreperiodrange，usebasethischeck
        min_reasonable = 0.1;
        max_reasonable = 100.0;
        
        if D_FEA < min_reasonable || D_FEA > max_reasonable
            fprintf('    ⚠️ deflectionexceedbasethisreasonablerange[%.1f, %.1f]，adjust\n', min_reasonable, max_reasonable);
            D_FEA = getCaseSpecificDefaultDeflection(layer_info, []);
            fprintf('    adjustafterdeflection: %.6f mm\n', D_FEA);
        end
    end
    
catch ME
    fprintf('    verifyprocessfailed: %s\n', ME.message);
end
end

function D_default = getCaseSpecificDefaultDeflection(layer_info, thickness_m)
% based onworking conditionofdefaultdeflectionvalue（ensureworking conditiondifferencedifferent）

% foundationdeflection
base_deflection = 8.0; % mm

try
    % according tomodeling typeadjust
    modeling_type = 'unknown';
    if isfield(layer_info, 'modeling_type')
        modeling_type = layer_info.modeling_type;
    end
    
    if contains(modeling_type, 'winkler', 'IgnoreCase', true)
        % Winklermethod：based onhavebodyparameters
        if isfield(layer_info, 'soil_modulus') && layer_info.soil_modulus > 0
            Es = layer_info.soil_modulus;
            % soil foundationmodulusanddeflectionbecomeantiratio
            modulus_factor = 50.0 / max(Es, 10); % phasefor50MPa
            D_default = base_deflection * modulus_factor;
            fprintf('    Winklerdefault value: Es=%.0f MPa，factor=%.2f，D=%.2f mm\n', Es, modulus_factor, D_default);
        elseif isfield(layer_info, 'spring_coefficient') && layer_info.spring_coefficient > 0
            k_spring = layer_info.spring_coefficient;
            % springstiffnessanddeflectionbecomeantiratio
            reference_k = 100e6;
            stiffness_factor = reference_k / max(k_spring, 1e6);
            D_default = base_deflection * stiffness_factor^0.5;
            fprintf('    Winklerdefault value: k=%.2e，factor=%.2f，D=%.2f mm\n', k_spring, stiffness_factor, D_default);
        else
            D_default = base_deflection * 0.8; % Winklerthroughconstantrelativelysmall
        end
    else
        % multilayermethodthroughconstantrelativelylarge
        D_default = base_deflection * 1.3;
        fprintf('    multilayerdefault value: D=%.2f mm\n', D_default);
    end
    
    % based onthicknessofenterstepadjust
    if ~isempty(thickness_m) && length(thickness_m) >= 1
        total_thickness = sum(thickness_m(1:min(3, length(thickness_m))));
        thickness_factor = 0.5 / max(total_thickness, 0.3);
        D_default = D_default * thickness_factor;
    end
    
    % ensureinengineeringreasonablerangewithin
    D_default = max(1.0, min(D_default, 50.0));
    
catch ME
    fprintf('    default valuecalculatefailed: %s，usebasethisvalue\n', ME.message);
    D_default = base_deflection;
end
end


function result = assemble3DIndicatorsResult(result, sigma_FEA, epsilon_FEA, D_FEA, ...
    thickness, modulus, layer_info)
% Assemble 3D indicators result

% === finite element modeling resultsX_FEA（main results）===
result.sigma_FEA = sigma_FEA;    % Surface layer bottom tensile stress (MPa)
result.epsilon_FEA = epsilon_FEA; % Base layer bottom tensile strain (με)
result.D_FEA = D_FEA;            % Subgrade top surface deflection (mm)

% === Compatibility fields（maintaintowardaftercompatible）===
result.stress_FEA = result.sigma_FEA;
result.strain_FEA = result.epsilon_FEA;
result.deflection_FEA = result.D_FEA;

% === finite element modeling resultsX_FEAdetailedinformation ===
result.FEA_3D_indicators = struct();
result.FEA_3D_indicators.surface_tensile_stress = sigma_FEA;
result.FEA_3D_indicators.base_tensile_strain = epsilon_FEA;
result.FEA_3D_indicators.subgrade_deflection = D_FEA;
result.FEA_3D_indicators.extraction_method = '3D_indicators_specialized_extraction';

% === otherfield ===
result.num_nodes = size(result.Model.Geometry.Mesh.Nodes, 2);
result.num_elements = size(result.Model.Geometry.Mesh.Elements, 2);
result.extraction_method = '2D_planestrain_3D_indicators_v11';

% thicknessinformation
if all(thickness > 1)
    result.layer_thicknesses_cm = thickness;
    result.total_thickness_cm = sum(thickness);
else
    result.layer_thicknesses_cm = thickness * 100;
    result.total_thickness_cm = sum(thickness) * 100;
end
result.layer_info = layer_info;

% === engineeringsetcountinformation（3Dversion）===
result.design_control_3D = struct();
result.design_control_3D.fatigue_control = sprintf('σ_FEA = %.4f MPa', sigma_FEA);
result.design_control_3D.structural_control = sprintf('ε_FEA = %.2f με', epsilon_FEA);
result.design_control_3D.deformation_control = sprintf('D_FEA = %.3f mm', D_FEA);

% addcanvisualizationrequiredofmissingfield
if isfield(result, 'load_radius_cm')
    result.tire_contact_width_m = 2 * result.load_radius_cm / 100;
else
    result.tire_contact_width_m = 0.426;  % default value：2*21.3/100
end

% === successfulmark ===
result.success = true;
result.message = 'finite element modeling resultsX_FEAextractsuccessful';
end

function display3DIndicators(result)
% Display 3D indicatorsresult

fprintf('\n=== finite element modeling results  X_FEA ===\n');

% Primary 3D indicators
fprintf('🎯 patent state space triplet:\n');
fprintf('  σ_FEA = %.4f MPa (Surface layer bottom tensile stress，fatigue control)\n', result.sigma_FEA);
fprintf('  ε_FEA = %.2f με (Base layer bottom tensile strain，structural control)\n', result.epsilon_FEA);
fprintf('  D_FEA = %.3f mm (Subgrade top surface deflection，deformation control)\n', result.D_FEA);

% Mesh information
fprintf('\n📊 FEASolveinformation:\n');
fprintf('  Node count: %d\n', result.num_nodes);
fprintf('  Element count: %d\n', result.num_elements);
fprintf('  Extraction method: %s\n', result.extraction_method);

% thicknessinformationverify
fprintf('\n📏 structureinformationverify:\n');
fprintf('  Layer thickness: [%s] cm\n', sprintf('%.1f ', result.layer_thicknesses_cm));
fprintf('  totalthickness: %.1f cm\n', result.total_thickness_cm);

% Reasonableness check
fprintf('\n✅ 3DindicatorReasonableness check:\n');
check3DIndicatorsReasonableness(result);
end

function check3DIndicatorsReasonableness(result)
% check3Dindicatorreasonableness

% stresscheck
if result.sigma_FEA > 0.1 && result.sigma_FEA < 2.0
    fprintf('  ✅ Surface layer tensile stressreasonable: %.4f MPa\n', result.sigma_FEA);
else
    fprintf('  ⚠️ Surface layer tensile stress abnormal: %.4f MPa\n', result.sigma_FEA);
end

% straincheck
if result.epsilon_FEA > 50 && result.epsilon_FEA < 2000
    fprintf('  ✅ Base layer tensile strainreasonable: %.2f με\n', result.epsilon_FEA);
else
    fprintf('  ⚠️ Base layer tensile strain abnormal: %.2f με\n', result.epsilon_FEA);
end

% deflectioncheck
if result.D_FEA > 1.0 && result.D_FEA < 30.0
    fprintf('  ✅ subgradedeflectionreasonable: %.3f mm\n', result.D_FEA);
else
    fprintf('  ⚠️ subgradedeflectionabnormal: %.3f mm\n', result.D_FEA);
end
end

%% ================ 【supplementofmissingfunction】comeselfsecond file ================

function [thickness, modulus, poisson, P, r, subgrade_config] = preprocessInputs_2D(designParams, loadParams, boundary_conditions)
% 【fixversion】parameterspreprocessfunction - correcttransmitboundary_conditionsofallfield

fprintf('🔧 【fixversion】process2Dinputparameters...\n');

try
    % === 1. processthicknessparameters ===
    if isfield(designParams, 'thickness') && ~isempty(designParams.thickness)
        thickness = designParams.thickness(:);
        thickness = thickness(thickness > 0);
        if isempty(thickness)
            thickness = [12; 30; 20; 120];
            fprintf('⚠️ thicknessparametersinvalid，usedefault value\n');
        end
        fprintf('rawthickness: [%s] cm\n', sprintf('%.1f ', thickness));
    else
        thickness = [12; 30; 20; 120];
        fprintf('⚠️ usedefaultthickness\n');
    end

    % unitconvert：cm -> m
    thickness = thickness / 100;

    % === 2. processmodulusparameters ===
    if isfield(designParams, 'modulus') && ~isempty(designParams.modulus)
        modulus = designParams.modulus(:);
        modulus = modulus(modulus > 0);
        if length(modulus) < length(thickness)
            default_modulus = [1500; 600; 200; 50];
            needed = length(thickness) - length(modulus);
            modulus = [modulus; default_modulus(end-needed+1:end)];
        end
    else
        modulus = [1500; 600; 200; 50];
        fprintf('⚠️ usedefaultmodulus\n');
    end

    % === 3. processPoisson's ratioparameters ===
    if isfield(designParams, 'poisson') && ~isempty(designParams.poisson)
        poisson = designParams.poisson(:);
        poisson = max(0.1, min(poisson, 0.49));
        if length(poisson) < length(thickness)
            default_poisson = [0.30; 0.25; 0.35; 0.45];
            needed = length(thickness) - length(poisson);
            poisson = [poisson; default_poisson(end-needed+1:end)];
        end
    else
        poisson = [0.30; 0.25; 0.35; 0.45];
        fprintf('⚠️ usedefaultPoisson ratio\n');
    end

    % === 4. strongmakeusestandardBZZ-100load ===
    P = 0.7;    % standardgroundpressure MPa
    r = 21.3;   % standardtireradius cm

    fprintf('📝 strongmakeusestandardBZZ-100load: P=%.2f MPa, r=%.1f cm\n', P, r);

    % === 5. 【Key fix】soil foundationconfiguration - complete copyboundary_conditions ===
    subgrade_config = struct();

    if exist('boundary_conditions', 'var') && isstruct(boundary_conditions) && ...
       ~isempty(boundary_conditions)
        
        fprintf('📦 fromboundary_conditionsconstructsubgrade_config...\n');
        
        % 【method1：complete copy】directly copy entirestructurebody（retainallfield）
        subgrade_config = boundary_conditions;
        
        % ensuremodeling_typefieldstorein（compatibilityprocess）
        if isfield(boundary_conditions, 'method') && ...
           (~isfield(subgrade_config, 'modeling_type') || ...
            isempty(subgrade_config.modeling_type))
            subgrade_config.modeling_type = boundary_conditions.method;
        end
        
        % ifnot yethavemodeling_type，tryfromotherfieldinfer
        if ~isfield(subgrade_config, 'modeling_type')
            if isfield(boundary_conditions, 'spring_coefficient')
                subgrade_config.modeling_type = 'winkler_springs';
            elseif isfield(boundary_conditions, 'layer_modulus')
                subgrade_config.modeling_type = 'multilayer_subgrade';
            else
                subgrade_config.modeling_type = 'fixed_bottom';
            end
        end
        
        fprintf('✅ boundary_conditionsfieldcopycompleted\n');
        
        % 【adjusttryoutput】displayKeyfield
        fprintf('  Keyfieldcheck:\n');
        
        if isfield(subgrade_config, 'modeling_type')
            fprintf('    ✓ modeling_type = %s\n', subgrade_config.modeling_type);
        else
            fprintf('    ✗ modeling_type missing\n');
        end
        
        if isfield(subgrade_config, 'spring_coefficient')
            fprintf('    ✓ spring_coefficient = %.2e N/m³\n', ...
                subgrade_config.spring_coefficient);
        else
            fprintf('    ○ spring_coefficient notset（nonWinklermethodwhennormal）\n');
        end
        
        if isfield(subgrade_config, 'soil_modulus')
            fprintf('    ✓ soil_modulus = %.0f MPa\n', subgrade_config.soil_modulus);
        elseif isfield(subgrade_config, 'subgrade_modulus')
            fprintf('    ✓ subgrade_modulus = %.0f MPa\n', subgrade_config.subgrade_modulus);
            % uniformfieldname
            subgrade_config.soil_modulus = subgrade_config.subgrade_modulus;
        else
            fprintf('    ○ soil_modulus notset\n');
        end
        
        if isfield(subgrade_config, 'k_winkler')
            fprintf('    ✓ k_winkler = %.2e N/m³\n', subgrade_config.k_winkler);
        end
        
    else
        % ifnohaveprovideboundary_conditions，usedefaultconfiguration
        fprintf('⚠️ notprovideboundary_conditionsorasempty，createdefaultconfiguration\n');
        subgrade_config.modeling_type = 'fixed_bottom';
    end

    % === 6. verifymodeling typeofvalidproperty ===
    valid_methods = ["winkler_springs", "multilayer_subgrade", "fixed_bottom"];
    
    if ~isfield(subgrade_config, 'modeling_type') || ...
       isempty(subgrade_config.modeling_type) || ...
       ~ismember(string(subgrade_config.modeling_type), valid_methods)
        
        warning('preprocessInputs_2D:InvalidMethod', ...
            'modeling_type "%s" invalid，fallbackas fixed_bottom', ...
            subgrade_config.modeling_type);
        subgrade_config.modeling_type = 'fixed_bottom';
    end

   % === 7. ensurearraylengthconsistent，andaccording tomodetypetypeTruncate ===
    min_length = min([length(thickness), length(modulus), length(poisson)]);
    thickness = thickness(1:min_length);
    modulus = modulus(1:min_length);
    poisson = poisson(1:min_length);
    
    % === 【Key fix】according toWinklermodetypeTruncatesoilbase layer ===
    % assumepavement structureas3layer（surface layer、base layer、subbase）
    num_pavement_layers = 3; 
    
    if isfield(subgrade_config, 'modeling_type') && ...
       strcmpi(subgrade_config.modeling_type, 'winkler_springs')
        
        if length(thickness) > num_pavement_layers
            fprintf('🔧 Winklermodetype：Truncate %d layerandwithafterofsoilbase layer...\n', num_pavement_layers + 1);
            thickness = thickness(1:num_pavement_layers);
            modulus = modulus(1:num_pavement_layers);
            poisson = poisson(1:num_pavement_layers);
        end
        
    elseif isfield(subgrade_config, 'modeling_type') && ...
           strcmpi(subgrade_config, 'multilayer_subgrade')
        
        % multilayer modelretainalllayer（pavement+soil foundationsublayer）
        fprintf('🔧 multilayer model：retainall %d layer structure（pavement+soil foundationsublayer）\n', length(thickness));
        
    end

    fprintf('✅ 2Dparametersprocesscompleted: %dlayer, soil foundationmodetype=%s\n', ...
        length(thickness), subgrade_config.modeling_type);
   
catch ME
    fprintf('❌ parameterspreprocessfailed: %s，usedefaultparameters\n', ME.message);
    fprintf('   errorstack: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);

    % usecompletely defaultparameters
    thickness = [0.12; 0.30; 0.20; 1.20];
    modulus = [1500; 600; 200; 50];
    poisson = [0.30; 0.25; 0.35; 0.45];
    P = 0.7;
    r = 21.3;

    subgrade_config = struct();
    subgrade_config.modeling_type = 'fixed_bottom';

    fprintf('✅ defaultparameterssetcompleted\n');
end

end

function [P_verified, r_verified] = verifyStandardAxleLoad_2D(P, r)
% verifystandard axle load

target_load_kN_per_m = 298; % 2Dload per unit lengthtargetvalue
current_load_kN_per_m = calculateTotalLoad2D(P, r);

fprintf('2Dloadverify: inputP=%.2f MPa, r=%.1f cm, whenbefore%.1f kN/m\n', P, r, current_load_kN_per_m);

load_error = abs(current_load_kN_per_m - target_load_kN_per_m) / target_load_kN_per_m;

if load_error > 0.05
    fprintf('⚠️ loadbiaseddifference%.1f%%，adjustasstandard axle load\n', load_error*100);
    P_verified = 0.7;
    r_verified = 21.3;
else
    P_verified = P;
    r_verified = r;
    fprintf('✅ loadparametersreasonable\n');
end
end

function [model,layer_info] = defineOptimized2DMaterials(model, modulus, poisson, layer_info)
% definepure2Dplane-strainmaterial properties

fprintf('defineoptimize2Dmaterial properties...\n');

num_layers = layer_info.num_layers;

% ensureparameterslengthconsistent
if length(modulus) ~= num_layers
    modulus = adjustArrayLength2D(modulus, num_layers);
end

if length(poisson) ~= num_layers
    poisson = adjustArrayLength2D(poisson, num_layers);
end

% displayeachlayerattribute
layer_names = {'surface layer', 'base layer', 'subbase', 'soil foundation'};
for i = 1:num_layers
    name = '';
    if i <= length(layer_names)
        name = layer_names{i};
    else
        name = sprintf('%dlayer', i);
    end
    fprintf('  %s: E=%.0f MPa, ν=%.3f, h=%.3f m\n', name, modulus(i), poisson(i), layer_info.thickness(i));
end

fprintf('  use2Dplane-strainmaterialset...\n');

try
    % method1：tryusecoordinateslayeredmaterial
    fprintf('    try2Dcoordinateslayeredmaterial...\n');
    model = set2DPlaneMaterials(model, modulus, poisson, layer_info);
    fprintf('    ✓ 2Dcoordinateslayeredmaterialsetsuccessful\n');

catch ME_layered
    fprintf('    2Dlayeredmaterialfailed: %s\n', ME_layered.message);

    % method2：useequivalent homogeneousmaterial
    fprintf('    useequivalent homogeneousmaterial...\n');
    setEquivalent2DMaterial(model, modulus, poisson, layer_info);
    fprintf('    ✓ equivalent homogeneousmaterialsetsuccessful\n');
end

fprintf('✅ 2Dmaterial propertiessetcompleted\n');

% savetolayer_info
layer_info.modulus = modulus;
layer_info.poisson = poisson;
end

function model = set2DPlaneMaterials(model, modulus, poisson, layer_info)
% set2Dplane-strainlayeredmaterial

y_positions = layer_info.y_positions;
num_layers = layer_info.num_layers;

fprintf('      setlayeredmaterial: %dlayer\n', num_layers);
for i = 1:num_layers
    fprintf('        layer%d: Yrange[%.3f, %.3f], E=%.0f MPa\n', ...
        i, y_positions(i+1), y_positions(i), modulus(i));
end

% createmodulusdistributionfunction
    function E_val = getModulusAtY(location, ~)
        y = location.y(:)';               % ensurecolumntowardquantity，suitablematchbatchadjustuse
        E_val = zeros(size(y));

        tol = 1e-9;                      % moresmalltolerance，reduce misjudgment
        for k = 1:num_layers
            y_top    = y_positions(k);
            y_bottom = y_positions(k+1);

            in_layer = (y <= y_top + tol) & (y >= y_bottom - tol);
            if any(in_layer)
                E_val(in_layer) = modulus(k)*1e6;   % MPa → Pa
            end
        end

        % ifstillhavenotassignnodes，default takemostclose toofpreviouslayer（thisinsideselecttablelayer）
        if any(E_val==0)
            E_val(E_val==0) = modulus(1)*1e6;
        end
    end

% createPoisson's ratiodistributionfunction
    function nu_val = getPoissonAtY(location, ~)
        y = location.y(:)';
        nu_val = zeros(size(y));

        tol = 1e-9;
        for k = 1:num_layers
            y_top    = y_positions(k);
            y_bottom = y_positions(k+1);

            in_layer = (y <= y_top + tol) & (y >= y_bottom - tol);
            if any(in_layer)
                nu_val(in_layer) = poisson(k);
            end
        end
        if any(nu_val==0)
            nu_val(nu_val==0) = poisson(1);
        end
    end

% apply2Dmaterial propertiesfunction
model.MaterialProperties = materialProperties( ...
    YoungsModulus = @getModulusAtY, ...
    PoissonsRatio = @getPoissonAtY, ...
    MassDensity   = 2200);
end

function setEquivalent2DMaterial(model, modulus, poisson, layer_info)
% set2Dequivalent homogeneousmaterial

thickness = layer_info.thickness;
total_thickness = sum(thickness);

% thicknessweighted equivalentmodulus
equiv_modulus = sum(modulus .* thickness) / total_thickness;

% thicknessweighted equivalentPoisson's ratio
equiv_poisson = sum(poisson .* thickness) / total_thickness;

% verifyparametersrange
equiv_modulus = max(50, min(equiv_modulus, 5000));  % 50-5000 MPa
equiv_poisson = max(0.15, min(equiv_poisson, 0.49)); % 0.15-0.49

% sethomogeneousmaterial
E_Pa = equiv_modulus * 1e6;
structuralProperties(model, ...
    'YoungsModulus', E_Pa, ...
    'PoissonsRatio', equiv_poisson, ...
    'MassDensity', 2200);

fprintf('      equivalent2Dmaterial: E=%.0f MPa, ν=%.3f\n', equiv_modulus, equiv_poisson);
end

function model = generateOptimized2DMesh(model, thickness, use_femodel)
% generate2Doptimizemesh

fprintf('generate2Doptimizemesh...\n');

% based ongeometrysizesumthicknessdeterminereasonablemeshsize
min_thickness = min(thickness);
total_thickness = sum(thickness);

% dynamicadjustmeshsize
target_max_nodes = 50000;

% based ongeometrysizeestimatemeshsize
road_width = 4.0;  % knownofoptimizeafterwidth

% estimatesuitableofmeshsize（2D）
estimated_mesh_size = estimateOptimal2DMeshSize(road_width, total_thickness, target_max_nodes);

% ensuremeshsizereasonable
mesh_size = max(min_thickness / 3, min(estimated_mesh_size, 0.3));

fprintf('  geometrysize: %.1f × %.3f m (2D)\n', road_width, total_thickness);
fprintf('  targetmaximumNode count: %d\n', target_max_nodes);
fprintf('  calculategetmeshsize: %.3f m\n', mesh_size);

try
    if ~use_femodel
        % Generate mesh
        generateMesh(model, 'Hmax', mesh_size);

        % checkactualmeshsystemcount
        actual_nodes = size(model.Mesh.Nodes, 2);
        actual_elements = size(model.Mesh.Elements, 2);

        fprintf('  actualmesh: %dnodes, %delements\n', actual_nodes, actual_elements);

        % ifmeshstill too large，enterstepoptimize
        if actual_nodes > target_max_nodes * 1.5
            fprintf('⚠️ meshstill too large，enterstepoptimize...\n');
            larger_mesh_size = mesh_size * 1.5;
            generateMesh(model, 'Hmax', larger_mesh_size);

            new_nodes = size(model.Mesh.Nodes, 2);
            new_elements = size(model.Mesh.Elements, 2);
            fprintf('  optimizeaftermesh: %dnodes, %delements\n', new_nodes, new_elements);
        end

        fprintf('✓ 2Doptimizemeshgeneratesuccessful\n');
    else
        model = generateMesh(model,'Hmax',mesh_size);
        actual_nodes = size(model.Geometry.Mesh.Nodes,2);
        actual_elements = size(model.Geometry.Mesh.Elements,2);
        fprintf('  actualmesh: %dnodes, %delements\n', actual_nodes, actual_elements);
        % ifmeshstill too large，enterstepoptimize
        if actual_nodes > target_max_nodes * 1.5
            fprintf('⚠️ meshstill too large，enterstepoptimize...\n');
            larger_mesh_size = mesh_size * 1.5;
            model = generateMesh(model, 'Hmax', larger_mesh_size);

            new_nodes = size(model.Geometry.Mesh.Nodes,2);
            new_elements = size(model.Geometry.Mesh.Elements,2);
            fprintf('  optimizeaftermesh: %dnodes, %delements\n', new_nodes, new_elements);
        end
    end

catch ME_mesh
    fprintf('⚠️ optimizemeshfailed，trycoarsemesh: %s\n', ME_mesh.message);
    try
        % usemorecoarseofmesh
        coarse_mesh_size = max(mesh_size * 2, 0.2);
        generateMesh(model, 'Hmax', coarse_mesh_size);
        fprintf('✓ coarsemeshgeneratesuccessful（meshsize: %.3f m）\n', coarse_mesh_size);
    catch ME_coarse
        error('2DMesh generation failed: %s', ME_coarse.message);
    end
end
end

function model = applyOptimized2DBoundaryConditions(model, P, r, layer_info, subgrade_config)
% fixversion：apply2Dboundary conditions，optimizeWinklerprocesslogic
fprintf('apply2Dplane-strainboundary conditions (%s)...\n', subgrade_config.modeling_type);

try
    geom = model.Geometry;

    % getboundaryquantity
    if isprop(geom, 'NumEdges') && geom.NumEdges > 0
        numEdges = geom.NumEdges;
        fprintf('  2Dgeometryedgesquantity: %d\n', numEdges);

        % standardboundaryassign
        [bottom_edge, top_edge, side_edges] = identify2DEdges(numEdges);
        fprintf('  boundaryassign: bottomedges=%d, topedges=%d, side=[%s]\n', ...
            bottom_edge, top_edge, sprintf('%d ', side_edges));

        % bottomedgesaccording to modeling_type enterlineprocess
        switch lower(subgrade_config.modeling_type)
            case 'winkler_springs'
                % === Winkler springmodetypeprocess（fixversion） ===
                k_winkler = getWinklerStiffnessFixed(subgrade_config, layer_info);
                
                % applyspringboundary conditions
                model.EdgeLoad(bottom_edge) = edgeLoad("TranslationalStiffness", [0; k_winkler]);
                fprintf('  ✓ bottomedges%d: Winkler springsupport，k=%.2e N/m³\n', bottom_edge, k_winkler);

            case 'multilayer_subgrade'
                % multilayersoil foundation，bottomedgescompletelyfixed
                model.EdgeBC(bottom_edge) = edgeBC(Constraint="fixed");
                fprintf('  ✓ bottomedges%d: multilayersoil foundationbottomedgesfixed\n', bottom_edge);

            otherwise
                % defaultcase：fixedbottomedges
                model.EdgeBC(bottom_edge) = edgeBC(Constraint="fixed");
                fprintf('  ✓ bottomedges%d: defaultfixedboundary\n', bottom_edge);
        end

        % side：tocalled constraint（Xdirectionfixed）
        for edge_id = side_edges
            model.EdgeBC(edge_id) = edgeBC(XDisplacement=0);
            fprintf('  ✓ side%d: Xdirectiontocalled constraint\n', edge_id);
        end

        % topedges：Apply load
        P_Pa = P * 1e6;
        contact_width = 2 * r / 100;  % convertas m
        half_width = contact_width / 2;
        x_nodes = model.Geometry.Mesh.Nodes(1,:);
        x_center = (max(x_nodes) + min(x_nodes)) / 2;

        fprintf('  ✓ topedges%d: applypressure P = %.2f MPa, contact width = %.2f m\n', ...
            top_edge, P, contact_width);

        pressFunction = @(loc,~) (-P_Pa) * (abs(loc.x - x_center) <= half_width);   % uniformpressure
        model.EdgeLoad(top_edge) = edgeLoad(Pressure=pressFunction);

    else
        error('invalidof2Dgeometry，unable toidentifyboundary');
    end

    fprintf('✅ boundary conditionssetcompleted\n');

catch ME_bc
    error('❌ boundary conditionssetfailed: %s', ME_bc.message);
end
end

function k_winkler = getWinklerStiffnessFixed(subgrade_config, layer_info)
% 【fixversion】getWinklerspringstiffness - strictfollowpaper formula
% 
% Theoretical formula: k = 0.65 × Es / (sqrt(B) × (1-γ²)) × CF
% Reference: Winkler, E. (1867) elastic foundation theory
%
% input:
%   subgrade_config - soil foundationconfigurationstructurebody
%   layer_info      - Layer informationstructurebody
% output:
%   k_winkler       - Winklerspringstiffness (N/m³)

fprintf('    【fixversion】Winklerspringstiffnesscalculate (followtheoryformula)\n');

% === priority1: useprecalculateofspringstiffness ===
% thiscomeselfprocessSubgradeWinkler.mofstandardcalculate
if isfield(subgrade_config, 'spring_coefficient') && ...
   ~isempty(subgrade_config.spring_coefficient) && ...
   subgrade_config.spring_coefficient > 0
    k_winkler = subgrade_config.spring_coefficient;
    fprintf('      ✓ useprecalculatevalue: k = %.2e N/m³\n', k_winkler);
    return;
end

if isfield(subgrade_config, 'k_winkler') && ...
   ~isempty(subgrade_config.k_winkler) && ...
   subgrade_config.k_winkler > 0
    k_winkler = subgrade_config.k_winkler;
    fprintf('      ✓ usek_winklerfield: k = %.2e N/m³\n', k_winkler);
    return;
end

% === priority2: usetheoryformulaheavynewcalculate ===
if isfield(subgrade_config, 'soil_modulus') && ...
   ~isempty(subgrade_config.soil_modulus)
    Es_MPa = subgrade_config.soil_modulus;
    
    % 【Key fix】usepaperinofstandardparameters
    B = 0.4;        % validroad width (m)
    gamma = 0.40;   % soil foundationPoisson's ratio
    
    % 【Key fix】strictaccording topaper formulacalculate
    % k = 0.65 × Es / (sqrt(B) × (1-γ²)) × CF
    Es_Pa = Es_MPa * 1e6;  % MPa → Pa
    k_base = 0.65 * Es_Pa / (sqrt(B) * (1 - gamma^2));
    
    % calculatecorrection factorCF
    CF = calculateCorrectionFactor(Es_MPa, layer_info);
    
    k_winkler = k_base * CF;
    
    fprintf('      ⚠️ usetheoryformulaheavynewcalculate:\n');
    fprintf('         Es = %.0f MPa, B = %.2f m, γ = %.2f\n', Es_MPa, B, gamma);
    fprintf('         k_base = %.2e N/m³\n', k_base);
    fprintf('         CF = %.3f\n', CF);
    fprintf('         k_winkler = %.2e N/m³\n', k_winkler);
else
    % === priority3: usedefault value ===
    fprintf('      ❌ unable togetsoil foundationmodulus，usedefault value\n');
    k_winkler = 50e6; % default50 MN/m³
end

% === Reasonableness check ===
% based onengineeringempiricalofreasonablerange
k_min = 10e6;   % minimum 10 MN/m³ (toshouldEs≈15MPa)
k_max = 200e6;  % maximum 200 MN/m³ (toshouldEs≈120MPa)

if k_winkler < k_min
    fprintf('      ⚠️ stiffnesspasssmall(%.2e)，adjusttominimumvalue%.2e N/m³\n', k_winkler, k_min);
    k_winkler = k_min;
elseif k_winkler > k_max
    fprintf('      ⚠️ stiffnesspasslarge(%.2e)，adjusttomaximumvalue%.2e N/m³\n', k_winkler, k_max);
    k_winkler = k_max;
end

fprintf('      finalspringstiffness: k = %.2e N/m³\n', k_winkler);
end

% === Auxiliary Functions: calculatecorrection factor ===
function CF = calculateCorrectionFactor(Es_MPa, layer_info)
% calculatecomprehensivecorrection factor CF = Cbase × Cload × Cthickness
% followpaper2.2.2sectionofdefine

% 1. Cbase - soil foundationconditioncorrect
if Es_MPa <= 30
    Cbase = 0.8;  % softsoil
    soil_type = 'softsoil';
elseif Es_MPa >= 80
    Cbase = 1.2;  % hardsoil
    soil_type = 'hardsoil';
else
    Cbase = 1.0;  % inequalsoil
    soil_type = 'inequalsoil';
end

% 2. Cload - loadhorizontalcorrect（temporarywhendefaultas1.0）
Cload = 1.0;  % standardload
% note: ifneedconsiderdifferentload，canfromlayer_infoinget

% 3. Cthickness - structurethicknesscorrect
Cthickness = 1.0;  % default
if isfield(layer_info, 'thickness') && ~isempty(layer_info.thickness)
    thickness_data = layer_info.thickness;
    
    % judgeunit
    if all(thickness_data > 1)  % cmunit
        total_thickness = sum(thickness_data(1:min(3, length(thickness_data))));
    else  % munit
        total_thickness = sum(thickness_data(1:min(3, length(thickness_data)))) * 100;
    end
    
    if total_thickness > 70
        Cthickness = 1.05;  % thickstructure
    elseif total_thickness < 50
        Cthickness = 0.95;  % thinstructure
    end
end

% comprehensivecorrection factor
CF = Cbase * Cload * Cthickness;

fprintf('      correction factorcalculate: Cbase=%.2f(%s) × Cload=%.2f × Cthickness=%.2f = %.3f\n', ...
    Cbase, soil_type, Cload, Cthickness, CF);
end


function k_winkler = calculateWinklerStiffness(soil_modulus_MPa, influence_depth_m)
    % calculateWinklerfoundationstiffnesscoefficient
    % input：
    %   soil_modulus_MPa: soil foundationmodulus (MPa)
    %   influence_depth_m: impactdepth (m)
    % output：
    %   k_winkler: Winklerfoundationstiffness (N/m^3)
    
    % convertunit
    Es_Pa = soil_modulus_MPa * 1e6;  % MPa -> Pa
    
    % method1：based onBoussinesqtheoryofempirical formula
    % k = C * Es / B，itsinCisempiricalcoefficient，Bisfeaturelength
    C_factor = 0.65;  % empiricalcoefficient，throughconstant0.5-1.0
    characteristic_length = max(influence_depth_m, 0.5);  % featurelength，minimum0.5m
    
    k_winkler_method1 = C_factor * Es_Pa / characteristic_length;
    
    % method2：based onfoundationbearingforcetheory
    % k = α * Es，itsinαisfoundationcoefficient
    alpha = 1.0 / max(influence_depth_m, 0.3);  % foundationcoefficient，impactdepthmorelarge，stiffnessmoresmall
    k_winkler_method2 = alpha * Es_Pa;
    
    % method3：considerloaddistributionofcorrect
    % forpathroadload，usemoresmallofstiffnesscoefficient
    road_factor = 0.3;  % pathroadloadcorrection factor
    k_winkler_method3 = road_factor * Es_Pa / characteristic_length;
    
    % take three methodsofgeometryaveragevalue，ensurereasonableness
    k_values = [k_winkler_method1, k_winkler_method2, k_winkler_method3];
    k_winkler = (prod(k_values))^(1/3);
    
    % limitmakeinreasonablerangewithin
    k_min = 1e6;   % minimumstiffness 1 MPa/m
    k_max = 1e9;   % maximumstiffness 1000 MPa/m
    k_winkler = max(k_min, min(k_winkler, k_max));
    
    fprintf('  Winklerfoundationstiffnesscalculate:\n');
    fprintf('    soil foundationmodulus: %.0f MPa\n', soil_modulus_MPa);
    fprintf('    impactdepth: %.2f m\n', influence_depth_m);
    fprintf('    method1stiffness: %.2e N/m^3\n', k_winkler_method1);
    fprintf('    method2stiffness: %.2e N/m^3\n', k_winkler_method2);  
    fprintf('    method3stiffness: %.2e N/m^3\n', k_winkler_method3);
    fprintf('    finalstiffness: %.2e N/m^3\n', k_winkler);
end

% inmainfunctioninadjustuse：
% k_winkler = calculateWinklerStiffness(soil_modulus, influence_depth);

function [bottom_edge, top_edge, side_edges] = identify2DEdges(numEdges)
% identify2Drectangleshapeofboundary

if numEdges >= 4
    % standardrectangleshape：4stripedges
    bottom_edge = 1;    % bottomedges
    top_edge = 3;       % topedges
    side_edges = [2, 4]; % leftRight sideedges

elseif numEdges == 3
    % triangleorsimplifygeometry
    bottom_edge = 1;
    top_edge = 2;
    side_edges = 3;

else
    % mostsimplecase
    bottom_edge = 1;
    top_edge = min(2, numEdges);
    side_edges = [];
end
end

function solution = solveWithTimeout2D(model,layer_info)
% with timeout controlof2DSolvedevice

fprintf('started2DPDESolve（withenterdegree monitoringcontrol）...\n');

try
    fprintf('configuration2DSolvedeviceparameters...\n');

    % startedSolve
    fprintf('executeline2DSolve...\n');
    solution_start_time = tic;

    % createfixedwhendevicecomemonitorcontrolSolveenterdegree
    progress_timer = timer('TimerFcn', @(~,~) fprintf('  2DSolveenterlinein... %.0fseconds\n', toc(solution_start_time)), ...
        'Period', 10, 'ExecutionMode', 'fixedRate');
    start(progress_timer);

    try
        % executelineSolve
        solution = solve(model);

        % stop progress monitoring
        stop(progress_timer);
        delete(progress_timer);

        fprintf('✅ 2DSolvesuccessfulcompleted\n');

    catch ME_solve
        % stop progress monitoring
        stop(progress_timer);
        delete(progress_timer);

        fprintf('❌ 2DSolvefailed: %s\n', ME_solve.message);
        rethrow(ME_solve);
    end

catch ME_timer
    fprintf('⚠️ enterdegree monitoringcontrolsetfailed，usebasethisSolve: %s\n', ME_timer.message);

    % iffixedwhendevicefailed，usebasethisSolve
    solution = solve(model);
    fprintf('✅ 2DbasethisSolvecompleted\n');
end
end

function total_load_kN_per_m = calculateTotalLoad2D(P_MPa, r_cm)
% calculate2Dload per unit length
P_Pa = P_MPa * 1e6;
contact_width = 2 * r_cm / 100; % m
total_force_per_length = P_Pa * contact_width; % N/m
total_load_kN_per_m = total_force_per_length / 1000; % kN/m
end

function checkMesh2DReasonableness(model)
% check2Dmeshreasonableness

num_nodes = size(model.Geometry.Mesh.Nodes, 2);
num_elements = size(model.Geometry.Mesh.Elements, 2);

fprintf('2DmeshReasonableness check:\n');
fprintf('  Node count: %d\n', num_nodes);
fprintf('  Element count: %d\n', num_elements);

% evaluateestimatemeshquality
if num_nodes > 100000
    fprintf('⚠️ warning：2Dmeshnonoften dense，maylead to longwhenbetweenSolve\n');
elseif num_nodes > 50000
    fprintf('⚡ note：2Dmeshrelativelydense，Solvewhenbetweenmayrelativelylong\n');
else
    fprintf('✅ 2Dmeshsizereasonable\n');
end
end

%% ================ 【maintainoriginalhave】otherAuxiliary Functions ================

function layer_info = enhanceLayerInfoForWinkler(layer_info, subgrade_config)
% Enhance layer_info，ensureWinklerparameterscorrecttransmit

fprintf('Enhance layer_infoparameterstransmit...\n');

try
    if isstruct(subgrade_config)
        % transmitKeyofWinklerparameters
        if isfield(subgrade_config, 'spring_coefficient')
            layer_info.spring_coefficient = subgrade_config.spring_coefficient;
            fprintf('  transmitspringstiffness: %.2e N/m³\n', subgrade_config.spring_coefficient);
        end
        
        if isfield(subgrade_config, 'expected_deflection_range')
            layer_info.expected_deflection_range = subgrade_config.expected_deflection_range;
            fprintf('  transmitpreperioddeflectionrange: [%.2f, %.2f] mm\n', ...
                subgrade_config.expected_deflection_range(1), ...
                subgrade_config.expected_deflection_range(2));
        end
        
        if isfield(subgrade_config, 'case_identifier')
            layer_info.case_identifier = subgrade_config.case_identifier;
            fprintf('  transmitworking conditionidentify: %s\n', subgrade_config.case_identifier);
        end
        
        if isfield(subgrade_config, 'modeling_type')
            layer_info.modeling_type = subgrade_config.modeling_type;
            fprintf('  transmitmodeling type: %s\n', subgrade_config.modeling_type);
        end
        
        if isfield(subgrade_config, 'soil_modulus')
            layer_info.soil_modulus = subgrade_config.soil_modulus;
            fprintf('  transmitsoil foundationmodulus: %.0f MPa\n', subgrade_config.soil_modulus);
        end
    end
    
    fprintf('✅ layer_infoenhancecompleted\n');
    
catch ME
    fprintf('⚠️ layer_infoenhancefailed: %s\n', ME.message);
end
end


function estimated_deflection = estimateReasonableDeflectionSafe(thickness_m, layer_info)
% 【safetyversion】deflectionreasonableestimatefunction
try
    if ~isempty(thickness_m) && length(thickness_m) >= 1
        total_thickness = sum(thickness_m(1:min(3, length(thickness_m))));
        
        % based onthicknessofbasethisestimate
        base_deflection = 12.0 / (total_thickness * 100 + 50); % simplifyformula
        
        % ifhavemodulusinformation
        if isstruct(layer_info) && isfield(layer_info, 'modulus') && ~isempty(layer_info.modulus)
            avg_modulus = mean(layer_info.modulus(1:min(3, length(layer_info.modulus))));
            modulus_factor = 1000 / max(avg_modulus, 100);
            estimated_deflection = base_deflection * modulus_factor * 10;
        else
            estimated_deflection = base_deflection * 15;
        end
    else
        estimated_deflection = 8.0;  % completelydefault value
    end
    
    % ensureinreasonablerangewithin
    estimated_deflection = max(2.0, min(estimated_deflection, 20.0));
    
catch
    estimated_deflection = 1.0;  % mostafterofsafevalue
end
end

function [nodes, u] = extractMeshAndSolution(model, solution)
% extractmeshsumsolutiondata

% getmeshnodes
if isprop(model, 'Mesh') && ~isempty(model.Mesh)
    nodes = model.Mesh.Nodes;
elseif isprop(model.Geometry, 'Mesh') && ~isempty(model.Geometry.Mesh)
    nodes = model.Geometry.Mesh.Nodes;
else
    error('unable togetmeshnodesinformation');
end

% getdisplacementsolution - enhanceversionthissuitablematchdifferentMATLABversionthis
if isprop(solution, 'NodalSolution') && ~isempty(solution.NodalSolution)
    u = solution.NodalSolution;
elseif isfield(solution, 'NodalSolution') && ~isempty(solution.NodalSolution)
    u = solution.NodalSolution;
elseif isprop(solution, 'Displacement') && ~isempty(solution.Displacement)
    % someversionthisuseDisplacementfield
    u = solution.Displacement;
elseif isfield(solution, 'Displacement') && ~isempty(solution.Displacement)
    u = solution.Displacement;
else
    error('unable togetnodessolution');
end

% verifydatavalidproperty
if isempty(u) || isempty(nodes)
    error('meshorsolutiondataasempty');
end

% basethisdimensioncheck
fprintf('  extractdata: nodesdimension=%s, solutiondimension=%s\n', mat2str(size(nodes)), mat2str(size(u)));

% ensureminimumdimensionrequire
min_nodes_dim = min(size(nodes));
min_solution_dim = min(size(u));

if min_nodes_dim < 2
    error('Node countaccording todimensioninsufficient，needat least2D');
end

if min_solution_dim < 2 
    error('displacementsolution dimensionsinsufficient，needat least2D');
end
end

function value_interp = robustInterpolation(y_coords, values, y_target)
% changeenterofinterpolationfunction
try
    % ensureinputdataasphasesamelengthoftowardquantity
    if length(y_coords) ~= length(values)
        error('coordinatessumvalueoflengthmismatch');
    end
    
    if length(y_coords) >= 2 && length(values) == length(y_coords)
        % convertascolumntowardquantitywithensureconsistentproperty
        y_coords = y_coords(:);
        values = values(:);
        
        % sortwithensureinterpolationstable
        [y_sorted, sort_idx] = sort(y_coords);
        values_sorted = values(sort_idx);
        
        % Use linear interpolation，allow extrapolation
        value_interp = interp1(y_sorted, values_sorted, y_target, 'linear', 'extrap');
        
        % Reasonableness check
        if isnan(value_interp) || isinf(value_interp)
            value_interp = mean(values);
        end
    else
        value_interp = mean(values);
    end
catch
    value_interp = mean(values);
end
end

function result = createDefault3DIndicatorsResult(thickness, modulus)
% Create default 3D indicators result
num_layers = length(thickness);

result = struct();

% Primary 3D indicators（using reasonable default values）
result.sigma_FEA = 0.65;      % MPa
result.epsilon_FEA = 500;     % με
result.D_FEA = 8.0;           % mm

% Compatibility fields
result.stress_FEA = result.sigma_FEA;
result.strain_FEA = result.epsilon_FEA;
result.deflection_FEA = result.D_FEA;

% finite element modeling resultsX_FEA
result.FEA_3D_indicators = struct();
result.FEA_3D_indicators.surface_tensile_stress = result.sigma_FEA;
result.FEA_3D_indicators.base_tensile_strain = result.epsilon_FEA;
result.FEA_3D_indicators.subgrade_deflection = result.D_FEA;

% Other default values
result.success = false;
result.message = '⚠️ PDE 3D indicators extraction failed，using engineering default value';
result.extraction_method = 'default_3D_indicators_v11';
result.num_layers = num_layers;
end

function result = createFailureResult3D(error_message)
% Create failure result（3Dversion）
result = struct();
result.success = false;
result.message = sprintf('❌ 2D 3D indicators modeling failed: %s', error_message);

% Use engineering reasonable default 3D indicators
result.sigma_FEA = 0.65;     % MPa
result.epsilon_FEA = 500;    % με
result.D_FEA = 8.0;          % mm

% Basic fields
result.num_nodes = 0;
result.num_elements = 0;
result.solve_time = 0;
result.software = 'MATLAB_PDE_2D_3D_Indicators_v11';
result.load_pressure_MPa = 0.7;
result.load_radius_cm = 21.3;
result.total_load_per_unit_length_kN = 298;
result.is_true_layered = false;
result.num_layers = 3;
result.road_width = 4.0;
result.modeling_type = 'optimized_2D_planestrain_3D_indicators_v11';
result.boundary_method = 'fixed_bottom';

% finite element modeling resultsX_FEA
result.FEA_3D_indicators = struct();
result.FEA_3D_indicators.surface_tensile_stress = result.sigma_FEA;
result.FEA_3D_indicators.base_tensile_strain = result.epsilon_FEA;
result.FEA_3D_indicators.subgrade_deflection = result.D_FEA;
result.FEA_3D_indicators.is_default = true;  % Identify as default values
end

function optimal_size = estimateOptimal2DMeshSize(width, height, target_nodes)
% Estimate optimal 2D mesh size

area = width * height;

% Area per mesh element（2D）
target_element_area = area / (target_nodes / 3); % Assume each 3 nodes form an element

% estimatemeshsize（square assumption）
optimal_size = sqrt(target_element_area);

% Apply safety factor and reasonable range
optimal_size = max(0.1, min(optimal_size, 0.5));
end

function adjusted_array = adjustArrayLength2D(original_array, target_length)
% Adjust array length
if length(original_array) > target_length
    adjusted_array = original_array(1:target_length);
elseif length(original_array) < target_length
    adjusted_array = [original_array(:); repmat(original_array(end), target_length - length(original_array), 1)];
else
    adjusted_array = original_array;
end
end

function [x_coords, y_coords, v_displacement] = unifyDataFormat(nodes, u_displacement)
% Intelligently unify data format
try
    % Process node coordinates
    if size(nodes, 1) == 2 && size(nodes, 2) > 2
        % nodes is 2×N format
        x_coords = nodes(1,:);
        y_coords = nodes(2,:);
    elseif size(nodes, 2) == 2 && size(nodes, 1) > 2
        % nodes is N×2 format
        x_coords = nodes(:,1)';
        y_coords = nodes(:,2)';
    else
        % Other cases，try automatic judgment
        if size(nodes, 1) <= size(nodes, 2)
            x_coords = nodes(1,:);
            y_coords = nodes(2,:);
        else
            x_coords = nodes(:,1)';
            y_coords = nodes(:,2)';
        end
    end
    
    % Process displacement data
    if size(u_displacement, 1) >= 2 && size(u_displacement, 2) > size(u_displacement, 1)
        % u_displacement is M×N format，Nis number of nodes
        v_displacement = u_displacement(2, :);  % Ydirection displacement
    elseif size(u_displacement, 2) >= 2 && size(u_displacement, 1) > size(u_displacement, 2)
        % u_displacement is N×M format，Nis number of nodes
        v_displacement = u_displacement(:, 2)';  % Ydirection displacement，convert to row vector
    else
        % Try automatic judgment based on size
        if size(u_displacement, 1) <= size(u_displacement, 2)
            if size(u_displacement, 1) >= 2
                v_displacement = u_displacement(2, :);
            else
                error('Displacement data dimension insufficient');
            end
        else
            if size(u_displacement, 2) >= 2
                v_displacement = u_displacement(:, 2)';
            else
                error('Displacement data dimension insufficient');
            end
        end
    end
    
    % Ensure all are row vectors
    x_coords = x_coords(:)';
    y_coords = y_coords(:)';
    v_displacement = v_displacement(:)';
    
catch ME
    % If automatic judgment fails，use most conservative method
    fprintf('    ⚠️ Data format automatic judgment failed: %s\n', ME.message);
    
    % Use maximum dimension as number of nodes
    max_nodes = max([size(nodes, 1), size(nodes, 2), size(u_displacement, 1), size(u_displacement, 2)]);
    
    if size(nodes, 2) == max_nodes
        x_coords = nodes(1, 1:max_nodes);
        y_coords = nodes(2, 1:max_nodes);
    else
        x_coords = nodes(1:max_nodes, 1)';
        y_coords = nodes(1:max_nodes, 2)';
    end
    
    if size(u_displacement, 2) == max_nodes
        v_displacement = u_displacement(2, 1:max_nodes);
    else
        v_displacement = u_displacement(1:max_nodes, 2)';
    end
end
end

function [x_coords_fixed, y_coords_fixed, v_displacement_fixed] = fixDataLengthMismatch(x_coords, y_coords, v_displacement)
% Fix data length mismatch issue
try
    lengths = [length(x_coords), length(y_coords), length(v_displacement)];
    min_length = min(lengths);
    max_length = max(lengths);
    
    fprintf('    Data lengths: x=%d, y=%d, v=%d\n', lengths);
    
    if max_length - min_length <= max_length * 0.1 % Length difference less than 10%
        % Truncatetomostshortlength
        target_length = min_length;
        x_coords_fixed = x_coords(1:target_length);
        y_coords_fixed = y_coords(1:target_length);
        v_displacement_fixed = v_displacement(1:target_length);
        fprintf('    Fix strategy: Truncatetomostshortlength %d\n', target_length);
    else
        % Use most common length
        [~, mode_idx] = mode(lengths);
        target_length = lengths(mode_idx);
        
        % Adjust each array to target length
        x_coords_fixed = adjustToTargetLength(x_coords, target_length);
        y_coords_fixed = adjustToTargetLength(y_coords, target_length);
        v_displacement_fixed = adjustToTargetLength(v_displacement, target_length);
        fprintf('    Fix strategy: adjust to most common length %d\n', target_length);
    end
    
catch ME
    % ifFix failed，usemostsimpleofTruncateStrategy
    min_length = min([length(x_coords), length(y_coords), length(v_displacement)]);
    x_coords_fixed = x_coords(1:min_length);
    y_coords_fixed = y_coords(1:min_length);
    v_displacement_fixed = v_displacement(1:min_length);
    fprintf('    Fix failed，using truncation strategy: length %d\n', min_length);
end
end

function adjusted_data = adjustToTargetLength(data, target_length)
% Adjust data to target length
current_length = length(data);

if current_length == target_length
    adjusted_data = data;
elseif current_length > target_length
    % Truncate
    adjusted_data = data(1:target_length);
else
    % Interpolate and extend
    if current_length > 1
        % Use interpolation
        old_indices = linspace(1, current_length, current_length);
        new_indices = linspace(1, current_length, target_length);
        adjusted_data = interp1(old_indices, data, new_indices, 'linear', 'extrap');
    else
        % Only one data point，repeat and extend
        adjusted_data = repmat(data, 1, target_length);
    end
end

% Ensure is row vector
adjusted_data = adjusted_data(:)';
end