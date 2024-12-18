$dirpath = $args[0]
$files = Get-ChildItem $dirpath
$wavfiles = $files | Where-Object { $_.Extension -eq ".wav" }

foreach ($wavfile in $wavfiles) {
    python .\separator.py $wavfile.FullName
    Write-Host "Separation done for $wavfile"
}