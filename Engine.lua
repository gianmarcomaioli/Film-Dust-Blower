-- Film Dust Blower: rendering, calling the engine, importing results.

local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrExportSession = import "LrExportSession"
local LrFileUtils = import "LrFileUtils"
local LrFunctionContext = import "LrFunctionContext"
local LrLogger = import "LrLogger"
local LrPathUtils = import "LrPathUtils"
local LrProgressScope = import "LrProgressScope"
local LrShell = import "LrShell"
local LrTasks = import "LrTasks"

local prefs = require "Prefs"

local logger = LrLogger("FilmDustBlower")
logger:enable("logfile")

local Engine = {}

local PYTHON_CANDIDATES_MAC = {
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
    "/usr/bin/python3",
}
local PYTHON_CANDIDATES_WIN = { "py -3", "python", "python3" }

-- ------------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------------

local function quote(s)
    return '"' .. s .. '"'
end

local function shell(cmd)
    -- cmd.exe /c strips one pair of outer quotes, so wrap the whole line.
    if WIN_ENV then
        return '"' .. cmd .. '"'
    end
    return cmd
end

local function pythonPrefix(cmd)
    if LrFileUtils.exists(cmd) then
        return quote(cmd)
    end
    return cmd
end

local function readText(path)
    local f = io.open(path, "r")
    if not f then
        return nil
    end
    local s = f:read("*a")
    f:close()
    return s
end

local function writeText(path, text)
    local f = assert(io.open(path, "w"))
    f:write(text)
    f:close()
end

local function scriptPath()
    return LrPathUtils.child(_PLUGIN.path, "dust_engine.py")
end

local function tempFile(name)
    return LrPathUtils.child(LrPathUtils.getStandardFilePath("temp"), name)
end

local function splitTabs(line)
    local out = {}
    for field in (line .. "\t"):gmatch("([^\t]*)\t") do
        out[#out + 1] = field
    end
    return out
end

-- ------------------------------------------------------------------------
-- Python discovery
-- ------------------------------------------------------------------------

local function tryPython(cmd)
    local out = tempFile("FilmDustBlower_check_" .. tostring(os.time()) .. ".txt")
    local line = pythonPrefix(cmd) .. " " .. quote(scriptPath()) .. " --check > " .. quote(out) .. " 2>&1"
    local status = LrTasks.execute(shell(line))
    local text = readText(out) or ""
    LrFileUtils.delete(out)
    return (status == 0) and (text:find("^ok") ~= nil), text
end

function Engine.findPython()
    local configured = prefs.pythonCommand or ""
    if configured ~= "" then
        local ok, text = tryPython(configured)
        if ok then
            return configured, text
        end
        return nil, "The Python command in the settings did not work:\n" .. configured ..
            "\n\nOutput:\n" .. text
    end
    local list = WIN_ENV and PYTHON_CANDIDATES_WIN or PYTHON_CANDIDATES_MAC
    local last, missing = "", nil
    for _, cmd in ipairs(list) do
        local ok, text = tryPython(cmd)
        if ok then
            prefs.pythonCommand = cmd
            return cmd, text
        end
        last = text
        missing = missing or text:match("^missing ([^\r\n(]+)")
    end
    if missing then
        missing = missing:gsub("%s+$", "")
        return nil, "Python was found, but these packages are missing: " .. missing .. "\n\n" ..
            "Install them with:\n    " .. (WIN_ENV and "py -m pip install " or "python3 -m pip install ") ..
            missing:gsub(",", "") .. "\n\nthen try again.\n\nDetails:\n" .. last
    end
    return nil, "Could not find Python 3 with NumPy, OpenCV and tifffile installed.\n\n" ..
        "Install them with:\n    " .. (WIN_ENV and "py -m pip install numpy opencv-python tifffile"
            or "python3 -m pip install numpy opencv-python tifffile") ..
        "\n\nthen, if needed, enter the full path to Python in Film Dust Blower: Settings.\n\n" ..
        "Last output:\n" .. last
end

-- ------------------------------------------------------------------------
-- Rendering one photo to a full-resolution 16-bit TIFF
-- ------------------------------------------------------------------------

-- Detection runs on sRGB, the colour space both detectors learned from; a
-- cleaned TIFF copy is rendered in ProPhoto RGB so no colours are clipped.
local function renderPhoto(photo, folder, colorSpace)
    local session = LrExportSession {
        photosToExport = { photo },
        exportSettings = {
            LR_format = "TIFF",
            LR_export_bitDepth = 16,
            LR_export_colorSpace = colorSpace or "sRGB",
            LR_tiff_compressionMethod = "compressionMethod_None",
            LR_size_doConstrain = false,
            LR_outputSharpeningOn = false,
            LR_export_destinationType = "specificFolder",
            LR_export_destinationPathPrefix = folder,
            LR_export_useSubfolder = false,
            LR_collisionHandling = "rename",
            LR_minimizeEmbeddedMetadata = true,
            LR_useWatermark = false,
            LR_reimportExportedPhoto = false,
            LR_includeVideoFiles = false,
        },
    }
    for _, rendition in session:renditions() do
        local ok, pathOrMsg = rendition:waitForRender()
        if ok then
            return pathOrMsg
        end
        return nil, tostring(pathOrMsg)
    end
    return nil, "nothing was rendered"
end

local function uniqueOutputPath(photo)
    local src = photo:getRawMetadata("path")
    local dir = LrPathUtils.parent(src)
    local base = LrPathUtils.removeExtension(LrPathUtils.leafName(src))
    local suffix = prefs.suffix or "-dust"
    local candidate = LrPathUtils.child(dir, base .. suffix .. ".tif")
    local n = 2
    while LrFileUtils.exists(candidate) do
        candidate = LrPathUtils.child(dir, base .. suffix .. "-" .. n .. ".tif")
        n = n + 1
    end
    return candidate
end

local function runEngine(python, workDir, line, dust)
    local jobPath = LrPathUtils.child(workDir, "job.txt")
    local outPath = LrPathUtils.child(workDir, "result.txt")
    local logPath = LrPathUtils.child(workDir, "python_log.txt")
    LrFileUtils.delete(outPath)
    writeText(jobPath, line .. "\n")
    local args = {
        pythonPrefix(python), quote(scriptPath()),
        "--job", quote(jobPath), "--out", quote(outPath),
        "--dust", dust,
        "--sensitivity", string.format("%.2f", tonumber(prefs.sensitivity) or 1),
        "--max-size", string.format("%.2f", tonumber(prefs.maxSize) or 1.5),
        "--max-spots", tostring(math.floor(tonumber(prefs.maxSpots) or 400)),
        "--verbose",
    }
    if not prefs.hairs then
        args[#args + 1] = "--no-hairs"
    end
    local cmd = table.concat(args, " ") .. " > " .. quote(logPath) .. " 2>&1"
    logger:info("running: " .. cmd)
    local status = LrTasks.execute(shell(cmd))
    local text = readText(outPath)
    if status ~= 0 or not text then
        return nil, "The dust engine failed (exit code " .. tostring(status) .. ").\n" .. (readText(logPath) or "")
    end
    local f = splitTabs((text:match("[^\r\n]+")) or "")
    if f[2] == "ok" then
        return { specks = tonumber(f[3]) or 0, hairs = tonumber(f[4]) or 0, note = f[7] or "" }
    end
    return nil, f[3] or "unknown error"
end

-- ------------------------------------------------------------------------
-- Develop-settings helpers
-- ------------------------------------------------------------------------

local function orientationOf(dev)
    local o = dev.orientation or dev.Orientation
    if type(o) == "string" and #o == 2 then
        return o
    end
    return "AB"
end

-- Converted negatives (Negative Lab Pro and similar) use inverted tone curves,
-- so dust shows light; on raw negatives and slides it shows dark.
local function curveInverted(t)
    if type(t) ~= "table" or #t < 4 then
        return false
    end
    return (tonumber(t[2]) or 0) > (tonumber(t[#t]) or 0)
end

local function dustPolarity(dev)
    local mode = prefs.dust or "auto"
    if mode == "dark" or mode == "bright" then
        return mode
    end
    if curveInverted(dev.ToneCurvePV2012Red) or curveInverted(dev.ToneCurvePV2012Green)
        or curveInverted(dev.ToneCurvePV2012Blue) or curveInverted(dev.ToneCurvePV2012) then
        return "bright"
    end
    return "dark"
end

local function randomHex(n)
    local digits = "0123456789ABCDEF"
    local out = {}
    for i = 1, n do
        local k = math.random(1, 16)
        out[i] = digits:sub(k, k)
    end
    return table.concat(out)
end

local function newMaskID()
    return randomHex(8) .. "-" .. randomHex(4) .. "-" .. randomHex(4) .. "-" .. randomHex(4) .. "-" .. randomHex(12)
end

-- One Lightroom heal spot, in the same form Lightroom 14 stores a brushed heal.
local function healArea(kind, radius, sx, sy, dabs)
    return {
        Feather = 0.135,
        HealVersion = 2,
        Masks = {
            {
                CenterWeight = 1,
                Dabs = dabs,
                Flow = 1,
                MaskActive = true,
                MaskBlendMode = 0,
                MaskID = newMaskID(),
                MaskInverted = false,
                MaskSyncID = randomHex(32),
                MaskValue = 1,
                Radius = radius,
                What = "Mask/Paint",
            },
        },
        Method = "gaussian",
        OffsetY = sy,
        Opacity = 1,
        Seed = 2,
        SourceState = "sourceAutoComputed",
        SourceX = sx,
        SpotType = "heal",
    }
end

local function readSpots(path)
    local areas = {}
    local text = readText(path) or ""
    for line in text:gmatch("[^\r\n]+") do
        local f = splitTabs(line)
        if f[1] == "spot" and f[6] then
            local dabs = {}
            for pair in (f[6] .. ";"):gmatch("([^;]+);") do
                local x, y = pair:match("^%s*(%S+)%s+(%S+)%s*$")
                x, y = tonumber(x), tonumber(y)
                if x and y then
                    dabs[#dabs + 1] = string.format("d %.6f %.6f", x, y)
                end
            end
            local r, sx, sy = tonumber(f[3]), tonumber(f[4]), tonumber(f[5])
            if #dabs > 0 and r and sx and sy then
                areas[#areas + 1] = healArea(f[2], r, sx, sy, dabs)
            end
        end
    end
    return areas
end

-- ------------------------------------------------------------------------
-- Heal-spot mode for one photo
-- ------------------------------------------------------------------------

local GEOMETRY_OFF = {
    CropLeft = 0, CropTop = 0, CropRight = 1, CropBottom = 1, CropAngle = 0,
    EnableLensCorrections = false, EnableTransform = false,
}

local function healPhoto(catalog, photo, python, workDir, mapPath)
    local dev = photo:getDevelopSettings()
    local saved = {
        CropLeft = dev.CropLeft or 0, CropTop = dev.CropTop or 0,
        CropRight = dev.CropRight or 1, CropBottom = dev.CropBottom or 1,
        CropAngle = dev.CropAngle or 0,
        EnableLensCorrections = dev.EnableLensCorrections ~= false,
        EnableTransform = dev.EnableTransform ~= false,
    }
    local existing = {}
    if type(dev.RetouchAreas) == "table" then
        for _, a in ipairs(dev.RetouchAreas) do
            existing[#existing + 1] = a
        end
    end
    local dust = dustPolarity(dev)
    local crop = string.format("%.6f,%.6f,%.6f,%.6f,%.6f", saved.CropLeft, saved.CropTop,
        saved.CropRight, saved.CropBottom, saved.CropAngle)

    -- render the whole frame without crop or lens geometry, so render pixels
    -- line up with the coordinates Lightroom uses for spots
    catalog:withWriteAccessDo("Film Dust Blower: prepare", function()
        photo:applyDevelopSettings(GEOMETRY_OFF)
    end, { timeout = 60 })

    local areas, note
    local ok, err = LrTasks.pcall(function()
        local renderDir = LrPathUtils.child(workDir, "render")
        LrFileUtils.createAllDirectories(renderDir)
        local render, rerr = renderPhoto(photo, renderDir)
        if not render then
            error(rerr or "render failed")
        end
        local spotsPath = LrPathUtils.child(workDir, "spots.txt")
        LrFileUtils.delete(spotsPath)
        local line = table.concat({ "1", render, "-", mapPath or "-", spotsPath, crop, orientationOf(dev) }, "\t")
        local res, msg = runEngine(python, workDir, line, dust)
        LrFileUtils.delete(render)
        if not res then
            error(msg)
        end
        note = res.note
        areas = readSpots(spotsPath)
    end)

    -- put the photo's own geometry back, adding the new spots in the same step
    local settings = {}
    for k, v in pairs(saved) do
        settings[k] = v
    end
    local added = 0
    if ok and areas and #areas > 0 then
        local merged = {}
        for _, a in ipairs(existing) do
            merged[#merged + 1] = a
        end
        for _, a in ipairs(areas) do
            merged[#merged + 1] = a
        end
        settings.RetouchAreas = merged
        added = #areas
    end
    catalog:withWriteAccessDo("Film Dust Blower", function()
        photo:applyDevelopSettings(settings, "Film Dust Blower")
    end, { timeout = 60 })

    if not ok then
        return nil, tostring(err)
    end
    -- check that Lightroom kept the spots
    local after = photo:getDevelopSettings().RetouchAreas
    local count = type(after) == "table" and #after or 0
    return { added = added, kept = count - #existing, dust = dust, note = note or "" }
end

-- ------------------------------------------------------------------------
-- Main command
-- mode: "preview" (dust maps only), "spots" (heal spots), "tiff" (cleaned copy)
-- ------------------------------------------------------------------------

local function process(context, mode)
    LrDialogs.attachErrorDialogToFunctionContext(context)
    local catalog = LrApplication.activeCatalog()
    math.randomseed(os.time())

    local photos = {}
    for _, photo in ipairs(catalog:getTargetPhotos()) do
        if photo:getRawMetadata("fileFormat") ~= "VIDEO" then
            photos[#photos + 1] = photo
        end
    end
    if #photos == 0 then
        LrDialogs.message("Film Dust Blower", "Select one or more scans first.", "info")
        return
    end

    local python, why = Engine.findPython()
    if not python then
        LrDialogs.message("Film Dust Blower: Python not found", why, "critical")
        return
    end

    local workDir = tempFile("FilmDustBlower_" .. tostring(os.time()))
    LrFileUtils.createAllDirectories(workDir)
    local mapDir = nil
    if mode == "preview" or prefs.keepMaps then
        mapDir = LrPathUtils.child(workDir, "dust maps")
        LrFileUtils.createAllDirectories(mapDir)
    end

    local progress = LrProgressScope {
        title = (mode == "preview") and "Film Dust Blower: preview" or "Film Dust Blower",
        functionContext = context,
    }

    local done, failed, summary, rejected = 0, {}, {}, 0
    for i, photo in ipairs(photos) do
        if progress:isCanceled() then
            break
        end
        local name = photo:getFormattedMetadata("fileName") or ("photo " .. i)
        progress:setCaption(string.format("%s (%d of %d)", name, i, #photos))
        progress:setPortionComplete(i - 1, #photos)
        local mapPath = "-"
        if mapDir then
            mapPath = LrPathUtils.child(mapDir, LrPathUtils.removeExtension(name) .. " dust map.jpg")
        end

        if mode == "spots" then
            local res, msg = healPhoto(catalog, photo, python, workDir, mapDir and mapPath or nil)
            if not res then
                failed[#failed + 1] = name .. ": " .. tostring(msg)
            else
                done = done + 1
                if res.kept < res.added then
                    rejected = rejected + 1
                    failed[#failed + 1] = string.format("%s: Lightroom kept %d of %d heal spots",
                        name, math.max(0, res.kept), res.added)
                end
                summary[#summary + 1] = string.format("%s: %d heal spots (dust shows %s)", name, res.added, res.dust)
            end
        else
            local dev = photo:getDevelopSettings()
            local dust = dustPolarity(dev)
            local renderDir = LrPathUtils.child(workDir, "render")
            LrFileUtils.createAllDirectories(renderDir)
            local render, err = renderPhoto(photo, renderDir, (mode == "tiff") and "ProPhotoRGB" or "sRGB")
            if not render then
                failed[#failed + 1] = name .. ": " .. tostring(err)
            else
                local outPath = (mode == "preview") and "-" or uniqueOutputPath(photo)
                local res, msg = runEngine(python, workDir, i .. "\t" .. render .. "\t" .. outPath .. "\t" .. mapPath, dust)
                LrFileUtils.delete(render)
                if not res then
                    failed[#failed + 1] = name .. ": " .. tostring(msg)
                else
                    summary[#summary + 1] = string.format("%s: %d specks, %d hairs (dust shows %s)",
                        name, res.specks, res.hairs, dust)
                    if mode == "tiff" then
                        catalog:withWriteAccessDo("Film Dust Blower: add cleaned copy", function()
                            catalog:addPhoto(outPath, photo, "above")
                        end, { timeout = 60 })
                    end
                    done = done + 1
                end
            end
        end
    end
    progress:done()
    LrFileUtils.delete(LrPathUtils.child(workDir, "render"))

    if mapDir and done > 0 then
        LrShell.revealInShell(mapDir)
    elseif not mapDir then
        LrFileUtils.delete(workDir)
    end

    local lines = {}
    for k = 1, math.min(#summary, 12) do
        lines[#lines + 1] = summary[k]
    end
    if #summary > 12 then
        lines[#lines + 1] = "... and " .. (#summary - 12) .. " more"
    end
    local head
    if mode == "preview" then
        head = string.format("Checked %d of %d photos. Nothing was changed.\n" ..
            "The dust maps show what would be removed: red circles for specks, blue boxes for hairs.", done, #photos)
    elseif mode == "spots" then
        head = string.format("Added heal spots to %d of %d photos.\n" ..
            "To undo, step back in the History panel to before \"Film Dust Blower: prepare\".", done, #photos)
    else
        head = string.format("Cleaned %d of %d photos. Each cleaned TIFF is stacked above its original.", done, #photos)
    end
    local text = head .. "\n\n" .. table.concat(lines, "\n")
    if #failed > 0 then
        text = text .. "\n\nProblems:\n" .. table.concat(failed, "\n")
    end
    if rejected > 0 then
        text = text .. "\n\nLightroom did not keep all the heal spots. Switch \"Repair as\" to " ..
            "\"Cleaned TIFF copy\" in the settings, and please report this."
    end
    LrDialogs.message("Film Dust Blower", text, #failed > 0 and "warning" or "info")
end

function Engine.run(previewOnly)
    LrFunctionContext.postAsyncTaskWithContext("FilmDustBlower", function(context)
        local mode = previewOnly and "preview" or ((prefs.repair == "tiff") and "tiff" or "spots")
        process(context, mode)
    end)
end

-- ------------------------------------------------------------------------
-- Diagnostics: dump develop settings (used to design a non-destructive
-- heal-spot mode later)
-- ------------------------------------------------------------------------

local function serialise(value, indent, depth, seen)
    indent = indent or ""
    depth = depth or 0
    seen = seen or {}
    local t = type(value)
    if t == "string" then
        return string.format("%q", value)
    elseif t == "number" or t == "boolean" or t == "nil" then
        return tostring(value)
    elseif t == "table" then
        if seen[value] then
            return "<cycle>"
        end
        if depth > 12 then
            return "<deep>"
        end
        seen[value] = true
        local keys = {}
        for k in pairs(value) do
            keys[#keys + 1] = k
        end
        table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
        local parts = {}
        for _, k in ipairs(keys) do
            parts[#parts + 1] = indent .. "  [" .. serialise(k) .. "] = " ..
                serialise(value[k], indent .. "  ", depth + 1, seen)
        end
        return "{\n" .. table.concat(parts, ",\n") .. "\n" .. indent .. "}"
    end
    return "<" .. t .. ">"
end

function Engine.dumpSettings()
    LrFunctionContext.postAsyncTaskWithContext("FilmDustBlowerDiag", function(context)
        LrDialogs.attachErrorDialogToFunctionContext(context)
        local catalog = LrApplication.activeCatalog()
        local photo = catalog:getTargetPhoto()
        if not photo then
            LrDialogs.message("Film Dust Blower", "Select a photo first.", "info")
            return
        end
        local name = photo:getFormattedMetadata("fileName") or "photo"
        local text = "-- Film Dust Blower diagnostics\n-- photo: " .. name ..
            "\n-- Lightroom: " .. tostring(LrApplication.versionString()) ..
            "\n\nreturn " .. serialise(photo:getDevelopSettings()) .. "\n"
        local dir = LrPathUtils.getStandardFilePath("desktop")
        local path = LrPathUtils.child(dir, "Film Dust Blower - develop settings - " ..
            LrPathUtils.removeExtension(name) .. ".txt")
        writeText(path, text)
        LrShell.revealInShell(path)
        LrDialogs.message("Film Dust Blower",
            "Saved the develop settings of " .. name .. " to your desktop.\n\n" ..
            "For this to be useful, add one or two spots by hand with Lightroom's Spot Removal tool " ..
            "(Heal mode) before running this command.", "info")
    end)
end

-- internal helpers exposed for testing
Engine._test = { readSpots = readSpots, dustPolarity = dustPolarity, healArea = healArea, curveInverted = curveInverted }

return Engine
