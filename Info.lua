--[[
Film Dust Blower - finds dust and hairs on film scans and repairs them.

Menu items appear under File > Plug-in Extras (available in every module).
]]

return {
    LrSdkVersion = 10.0,
    LrSdkMinimumVersion = 6.0,

    LrToolkitIdentifier = "com.filmdustblower.lightroom",
    LrPluginName = "Film Dust Blower",

    LrExportMenuItems = {
        {
            title = "Film Dust Blower: Preview Dust on Selected...",
            file = "PreviewSelected.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = "Film Dust Blower: Clean Selected Photos",
            file = "CleanSelected.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = "Film Dust Blower: Settings...",
            file = "SettingsDialog.lua",
        },
        {
            title = "Film Dust Blower: Save Develop Settings for Diagnosis",
            file = "Diagnostics.lua",
            enabledWhen = "photosSelected",
        },
    },

    LrPluginInfoProvider = "PluginInfo.lua",

    VERSION = { major = 1, minor = 4, revision = 0 },
}
