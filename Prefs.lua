-- Plugin preferences with defaults.

local LrPrefs = import "LrPrefs"

local prefs = LrPrefs.prefsForPlugin()

local defaults = {
    pythonCommand = "",     -- empty = find automatically
    repair = "spots",       -- spots: heal spots on the photo | tiff: separate cleaned TIFF
    dust = "auto",          -- auto | dark (raw negatives, slides) | bright (converted negatives)
    sensitivity = 1.0,      -- higher finds fainter dust but risks touching real detail
    maxSize = 1.5,          -- largest speck, percent of the short side
    hairs = true,           -- also remove hairs and fibres
    maxSpots = 400,         -- most heal spots added to one photo
    suffix = "-dust",       -- added to the cleaned TIFF's name
    keepMaps = false,       -- when cleaning, also save dust maps of what was removed
}

for k, v in pairs(defaults) do
    if prefs[k] == nil then
        prefs[k] = v
    end
end

-- version 1.0 introduced heal spots and automatic dust polarity
if (tonumber(prefs.prefsVersion) or 0) < 2 then
    prefs.repair = "spots"
    prefs.dust = "auto"
    prefs.prefsVersion = 2
end

return prefs
