function [pavement_params, subgrade_params] = separatePavementAndSubgrade(full_params)
% Separate pavement structure and subgrade parameters
% Input: full_params - Complete parameter structure
% Output: pavement_params - Pavement structure parameters (first 3 layers)
%         subgrade_params - Subgrade parameters (4th layer and beyond)

fprintf('Separating pavement structure and subgrade parameters...\n');

try
    % Validate input parameters
    if ~isstruct(full_params)
        error('Input parameter must be a structure');
    end
    
    % Display structure information
    fprintf('=== Input Parameter Diagnostics ===\n');
    
    % Safely get field names
    try
        field_names = fieldnames(full_params);
        fprintf('Detected %d fields\n', length(field_names));
        
        % Display fields individually
        fprintf('Field list:\n');
        for i = 1:length(field_names)
            fprintf('  [%d] %s\n', i, field_names{i});
        end
    catch field_error
        fprintf('Failed to get field names: %s\n', field_error.message);
        error('Cannot retrieve structure field information');
    end
    
    % Safe field checking
    thickness_field = '';
    modulus_field = '';
    poisson_field = '';
    
    % Check each field individually
    for i = 1:length(field_names)
        field_name = field_names{i};
        
        % Check thickness field
        if strcmp(field_name, 'thickness') || strcmp(field_name, 'Thickness') || strcmp(field_name, 'h')
            thickness_field = field_name;
            fprintf('Found thickness field: %s\n', thickness_field);
        end
        
        % Check modulus field
        if strcmp(field_name, 'modulus') || strcmp(field_name, 'Modulus') || strcmp(field_name, 'E')
            modulus_field = field_name;
            fprintf('Found modulus field: %s\n', modulus_field);
        end
        
        % Check Poisson's ratio field
        if strcmp(field_name, 'poisson') || strcmp(field_name, 'Poisson') || ...
           strcmp(field_name, 'nu') || strcmp(field_name, 'Nu') || strcmp(field_name, 'v')
            poisson_field = field_name;
            fprintf('Found Poisson field: %s\n', poisson_field);
        end
    end
    
    % Verify required fields
    if isempty(thickness_field)
        fprintf('Thickness field not found\n');
        error('Cannot find thickness field');
    end
    if isempty(modulus_field)
        fprintf('Modulus field not found\n');
        error('Cannot find modulus field');
    end
    if isempty(poisson_field)
        fprintf('Poisson field not found\n');
        error('Cannot find Poisson field');
    end
    
    fprintf('===================================\n');
    
    % Safely access parameter data
    try
        thickness_data = full_params.(thickness_field);
        modulus_data = full_params.(modulus_field);
        poisson_data = full_params.(poisson_field);
        
        % Ensure data is numeric
        if ~isnumeric(thickness_data) || ~isnumeric(modulus_data) || ~isnumeric(poisson_data)
            error('Parameter data must be numeric');
        end
        
        % Convert to column vectors
        thickness_data = thickness_data(:);
        modulus_data = modulus_data(:);
        poisson_data = poisson_data(:);
        
    catch access_error
        fprintf('Parameter access failed: %s\n', access_error.message);
        error('Cannot access parameter data');
    end
    
    % Get parameter length
    total_layers = length(thickness_data);
    pavement_layers = 3; % Number of pavement structure layers (surface, base, subbase)
    
    fprintf('  Total layers: %d, Pavement layers: %d\n', total_layers, pavement_layers);
    
    % Check parameter array length consistency
    modulus_len = length(modulus_data);
    poisson_len = length(poisson_data);
    
    fprintf('  Parameter lengths: thickness=%d, modulus=%d, poisson=%d\n', total_layers, modulus_len, poisson_len);
    
    if total_layers ~= modulus_len || total_layers ~= poisson_len
        fprintf('  Parameter array lengths inconsistent, aligning\n');
        
        % Align to minimum length
        min_len = min([total_layers, modulus_len, poisson_len]);
        if min_len < 3
            error('Valid parameter layers less than 3, cannot construct pavement structure');
        end
        
        thickness_data = thickness_data(1:min_len);
        modulus_data = modulus_data(1:min_len);
        poisson_data = poisson_data(1:min_len);
        total_layers = min_len;
        
        fprintf('    Aligned layer count: %d\n', total_layers);
    end
    
    % === 1. Separate pavement structure parameters (first 3 layers) ===
    pavement_params = struct();
    
    % Ensure pavement layers don't exceed total layers
    actual_pavement_layers = min(pavement_layers, total_layers);
    
    % Safe array indexing
    try
        pavement_params.thickness = thickness_data(1:actual_pavement_layers);
        pavement_params.modulus = modulus_data(1:actual_pavement_layers);
        pavement_params.poisson = poisson_data(1:actual_pavement_layers);
    catch index_error
        fprintf('Pavement parameter indexing failed: %s\n', index_error.message);
        error('Index out of bounds during pavement parameter separation');
    end
    
    % Safely copy other parameters
    safe_fields_to_copy = {'traffic_level', 'road_type', 'vehicle_speed_kmh', 'load_pressure', 'load_radius'};
    for i = 1:length(safe_fields_to_copy)
        field = safe_fields_to_copy{i};
        if isfield(full_params, field)
            try
                pavement_params.(field) = full_params.(field);
            catch copy_error
                fprintf('Failed to copy field %s: %s\n', field, copy_error.message);
            end
        end
    end
    
    % Safely copy material information
    if isfield(full_params, 'material')
        try
            material_data = full_params.material;
            if length(material_data) >= actual_pavement_layers
                pavement_params.material = material_data(1:actual_pavement_layers);
            end
        catch material_error
            fprintf('Failed to copy material information: %s\n', material_error.message);
        end
    end
    
    fprintf('  Pavement structure parameter separation complete: %d layers\n', length(pavement_params.thickness));
    
    % Display parameters safely
    fprintf('    Thickness: ');
    for i = 1:length(pavement_params.thickness)
        fprintf('%.1f ', pavement_params.thickness(i));
    end
    fprintf('cm\n');
    
    fprintf('    Modulus: ');
    for i = 1:length(pavement_params.modulus)
        fprintf('%.0f ', pavement_params.modulus(i));
    end
    fprintf('MPa\n');
    
    % === 2. Separate subgrade parameters (4th layer and beyond) ===
    subgrade_params = struct();
    
    if total_layers > actual_pavement_layers
        % Subgrade layers exist
        subgrade_start = actual_pavement_layers + 1;
        
        try
            subgrade_params.thickness = thickness_data(subgrade_start:end);
            subgrade_params.modulus = modulus_data(subgrade_start:end);
            subgrade_params.poisson = poisson_data(subgrade_start:end);
            subgrade_params.num_layers = length(subgrade_params.thickness);
        catch subgrade_index_error
            fprintf('Subgrade parameter indexing failed: %s\n', subgrade_index_error.message);
            error('Index out of bounds during subgrade parameter separation');
        end
        
        % Safely copy subgrade-related parameters
        subgrade_safe_fields = {'subgrade_type', 'subgrade_treatment', 'subgrade_modeling', 'boundary_method'};
        for i = 1:length(subgrade_safe_fields)
            field = subgrade_safe_fields{i};
            if isfield(full_params, field)
                try
                    subgrade_params.(field) = full_params.(field);
                catch subgrade_copy_error
                    fprintf('Failed to copy subgrade field %s: %s\n', field, subgrade_copy_error.message);
                end
            end
        end
        
        % Set default subgrade modeling type
        if ~isfield(subgrade_params, 'modeling_type')
            if isfield(full_params, 'subgrade_modeling')
                subgrade_params.modeling_type = full_params.subgrade_modeling;
            elseif isfield(full_params, 'boundary_method')
                subgrade_params.modeling_type = full_params.boundary_method;
            else
                subgrade_params.modeling_type = 'multilayer_subgrade';
            end
        end
        
        fprintf('  Subgrade parameter separation complete: %d layers\n', subgrade_params.num_layers);
        fprintf('    Modeling type: %s\n', subgrade_params.modeling_type);
    else
        % No subgrade layers
        subgrade_params.thickness = [];
        subgrade_params.modulus = [];
        subgrade_params.poisson = [];
        subgrade_params.num_layers = 0;
        subgrade_params.modeling_type = 'no_subgrade';
        
        fprintf('  No subgrade layers detected\n');
    end
    
    fprintf('Parameter separation complete\n');
    
catch ME
    fprintf('Parameter separation failed: %s\n', ME.message);
    
    % Provide safe debugging information
    fprintf('\n=== Safe Debugging Information ===\n');
    if exist('full_params', 'var') && isstruct(full_params)
        fprintf('Input parameter is a valid structure\n');
        
        try
            field_names = fieldnames(full_params);
            fprintf('Number of fields: %d\n', length(field_names));
            
            for i = 1:length(field_names)
                field = field_names{i};
                fprintf('Field[%d]: %s\n', i, field);
                
                try
                    value = full_params.(field);
                    if isnumeric(value)
                        fprintf('  Type: numeric, Length: %d\n', length(value));
                        if length(value) <= 10
                            fprintf('  Value: ');
                            for j = 1:length(value)
                                fprintf('%.2f ', value(j));
                            end
                            fprintf('\n');
                        end
                    else
                        fprintf('  Type: %s\n', class(value));
                    end
                catch field_access_error
                    fprintf('  Access failed: %s\n', field_access_error.message);
                end
            end
        catch debug_error
            fprintf('Failed to retrieve debug information: %s\n', debug_error.message);
        end
    else
        fprintf('Input parameter is not a valid structure\n');
    end
    fprintf('==================================\n');
    
    % Rethrow error
    error('Parameter separation failed: %s', ME.message);
end
end