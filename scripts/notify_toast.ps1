# Windows toast notifier for tranche_watch.py. Requires Windows PowerShell 5.1
# (powershell.exe) — WinRT type loading used here does not work under pwsh 7.
param(
    [string]$Title = "EdgeStack Tranche Watch",
    [string]$Body = ""
)
try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $escTitle = [System.Security.SecurityElement]::Escape($Title)
    $escBody = [System.Security.SecurityElement]::Escape($Body)
    $template = "<toast scenario='reminder'><visual><binding template='ToastGeneric'><text>$escTitle</text><text>$escBody</text></binding></visual></toast>"
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml($template)
    $toast = New-Object Windows.UI.Notifications.ToastNotification $xml
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("EdgeStack Tranche Watch").Show($toast)
    exit 0
} catch {
    Write-Error $_
    exit 1
}
