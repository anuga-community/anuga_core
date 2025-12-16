
function s = anuga_read_sww(filename)
%READ_ANUGA_SWW  Read ANUGA .sww output into a MATLAB struct.
%
%   s = ANUGA_READ_SWW('runup.sww')
%
% The returned struct 's' contains:
%   s.time           : time vector [nt]
%   s.x              : node x-coordinates [nn]
%   s.y              : node y-coordinates [nn]
%   s.volumes        : cell connectivity (triangles) [ne x 3] (1-based)
%   s.origin         : [xllcorner, yllcorner] if present
%   s.quantities     : list of quantity base names (no _c suffix)
%   s.<qname>_c      : centroid-based quantity array [nt x ne]
%
% Centroid-based variable names are kept with '_c' suffix, mirroring ANUGA.

% --- basic info
info      = ncinfo(filename);
varNames  = {info.Variables.Name};
attNames  = {info.Attributes.Name};


% --- helper to test existence
hasVar = @(name) any(strcmp(name, varNames));
attId  = @(name) find(strcmp(attNames, name), 1);

% --- Attributes
% --- starttime
starttimeID = attId('starttime');
if ~isempty(starttimeID)
    s.starttime = info.Attributes(starttimeID).Value;
end

% --- xllcorner
xllcornerID = attId('xllcorner');
if ~isempty(xllcornerID)
    s.xllcorner = info.Attributes(xllcornerID).Value;
end

% --- yllcorner
yllcornerID = attId('yllcorner');
if ~isempty(yllcornerID)
    s.yllcorner = info.Attributes(yllcornerID).Value;
end

% --- zone
zoneID = attId('zone');
if ~isempty(zoneID)
    s.zone = info.Attributes(zoneID).Value;
end

hemisphereID = attId('hemisphere');
if ~isempty(hemisphereID)
    s.hemisphere = info.Attributes(hemisphereID).Value;
end

% --- time
if hasVar('time')
    s.time = ncread(filename, 'time');       % [nt]
else
    error('Variable "time" not found in SWW file.');
end

% --- mesh: node coordinates
if hasVar('x') && hasVar('y')
    s.x = ncread(filename, 'x');            % [nn]
    s.y = ncread(filename, 'y');            % [nn]
else
    error('Variables "x" and/or "y" not found in SWW file.');
end

% --- mesh: connectivity (cells / triangles)
% ANUGA stores "volumes" as 0-based node indices; convert to MATLAB 1-based.
if hasVar('volumes')
    vols = ncread(filename, 'volumes');     % [3 x ne] or [ne x 3]
    vols = double(vols);
    % Make sure dimensions are [ne x 3]
    if size(vols,1) == 3 && size(vols,2) ~= 3
        vols = vols.';  % transpose to [ne x 3]
    end
    % Convert from 0-based to 1-based indices
    s.volumes = vols + 1;
else
    error('Variable "volumes" not found in SWW file.');
end



% --- detect centroid-based quantities (ending in "_c")
isCentroid = @(name) numel(name) > 2 && endsWith(name, '_c');

centroidVars = varNames(cellfun(isCentroid, varNames));

% store list of quantity base names (without _c suffix)
s.quantities = cellfun(@(n) n(1:end-2), centroidVars, 'UniformOutput', false);

% --- read centroid-based variables
for k = 1:numel(centroidVars)
    vname = centroidVars{k};
    data  = ncread(filename, vname);  % ANUGA convention: [nt x ne]
    % Some NetCDF writers may store as [ne x nt]; ensure [nt x ne]
    if size(data,1) ~= numel(s.time) && size(data,2) == numel(s.time)
        data = data.';
    end
    s.(vname) = data;
end

end
