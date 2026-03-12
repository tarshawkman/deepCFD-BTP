function [XB, YB] = LOAD_AIRFOIL_SELIG(fileName)
    % LOAD_AIRFOIL_SELIG Loads Selig-format airfoil coordinate data from a file.
    %
    % Inputs:
    %   fileName - Name of the airfoil file (without the .dat extension)
    %
    % Outputs:
    %   XB - Boundary point X-coordinates
    %   YB - Boundary point Y-coordinates

    % Construct full file path
    fullPath = [fileName '.dat'];
    
    % If file not found in current directory, check Airfoil_DAT_Selig folder
    if ~exist(fullPath, 'file')
        fullPath = fullPath;
        if ~exist(fullPath, 'file')
            error(['Airfoil file ' fileName '.dat not found in current directory']);
        end
    end

    % Open the file
    fid = fopen(fullPath, 'r');
    if fid == -1
        error(['Could not open file ' fullPath]);
    end

    % Read data skipping the first line (header)
    data = textscan(fid, '%f %f', 'HeaderLines', 1, 'CollectOutput', 1);
    
    % Close the file
    fclose(fid);

    % Extract boundary points
    XB = data{1}(:,1);
    YB = data{1}(:,2);
end
