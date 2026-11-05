-- File > Plug-in Extras > Film Dust Blower: Settings...

local LrDialogs = import "LrDialogs"
local LrFunctionContext = import "LrFunctionContext"
local LrView = import "LrView"

local SettingsUI = require "SettingsUI"

LrFunctionContext.postAsyncTaskWithContext("FilmDustBlowerSettings", function(context)
    LrDialogs.attachErrorDialogToFunctionContext(context)
    local f = LrView.osFactory()
    LrDialogs.presentModalDialog {
        title = "Film Dust Blower Settings",
        contents = SettingsUI.build(f),
        actionVerb = "Done",
        cancelVerb = "< exclude >",
    }
end)
