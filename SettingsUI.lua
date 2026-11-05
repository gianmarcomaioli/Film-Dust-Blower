-- Settings controls, shared by the Settings dialog and the Plug-in Manager panel.

local LrDialogs = import "LrDialogs"
local LrTasks = import "LrTasks"
local LrView = import "LrView"

local prefs = require "Prefs"
local Engine = require "Engine"

local bind = LrView.bind
local share = LrView.share

local SettingsUI = {}

local function label(f, text)
    return f:static_text { title = text, alignment = "right", width = share "fdb_label" }
end

function SettingsUI.build(f)
    return f:column {
        bind_to_object = prefs,
        spacing = f:control_spacing(),

        f:row {
            label(f, "Repair as:"),
            f:popup_menu {
                value = bind "repair",
                items = {
                    { title = "Heal spots on the photo (undo in History)", value = "spots" },
                    { title = "Separate cleaned TIFF copy", value = "tiff" },
                },
            },
        },

        f:row {
            label(f, "Dust shows:"),
            f:popup_menu {
                value = bind "dust",
                items = {
                    { title = "Automatic (light if the negative is already converted)", value = "auto" },
                    { title = "Dark (raw negatives, slides)", value = "dark" },
                    { title = "Light (converted negatives)", value = "bright" },
                },
            },
        },

        f:row {
            label(f, "Sensitivity:"),
            f:slider { value = bind "sensitivity", min = 0.5, max = 2, width = 160 },
            f:edit_field { value = bind "sensitivity", min = 0.3, max = 3, precision = 2, width_in_digits = 4 },
            f:static_text { title = "higher finds fainter dust, but may touch real detail" },
        },

        f:row {
            label(f, "Largest speck:"),
            f:slider { value = bind "maxSize", min = 0.5, max = 4, width = 160 },
            f:edit_field { value = bind "maxSize", min = 0.2, max = 8, precision = 1, width_in_digits = 4 },
            f:static_text { title = "% of the short side" },
        },

        f:row {
            label(f, ""),
            f:checkbox { title = "Remove hairs and fibres", value = bind "hairs" },
        },

        f:separator { fill_horizontal = 1 },

        f:row {
            label(f, "Most spots per photo:"),
            f:edit_field { value = bind "maxSpots", min = 10, max = 2000, precision = 0, width_in_digits = 5 },
            f:static_text { title = "the strongest are kept; many spots slow Lightroom down" },
        },
        f:row {
            label(f, "Cleaned TIFF name:"),
            f:static_text { title = "original name +" },
            f:edit_field { value = bind "suffix", width_in_chars = 10 },
            f:static_text { title = ".tif, saved next to the original (TIFF mode)" },
        },
        f:row {
            label(f, ""),
            f:checkbox { title = "When cleaning, also save dust maps of what was removed", value = bind "keepMaps" },
        },

        f:separator { fill_horizontal = 1 },

        f:row {
            label(f, "Python command:"),
            f:edit_field { value = bind "pythonCommand", width_in_chars = 34, immediate = true },
            f:push_button {
                title = "Test",
                action = function()
                    LrTasks.startAsyncTask(function()
                        local cmd, text = Engine.findPython()
                        if cmd then
                            LrDialogs.message("Python is working", "Using: " .. cmd .. "\n\n" .. text, "info")
                        else
                            LrDialogs.message("Python problem", text, "critical")
                        end
                    end)
                end,
            },
        },
        f:row {
            label(f, ""),
            f:static_text {
                title = "Leave empty to find Python automatically. Needs numpy, opencv-python and tifffile.",
                size = "small",
            },
        },
    }
end

return SettingsUI
