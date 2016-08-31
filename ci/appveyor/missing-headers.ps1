function InstallMissingHeaders () {
    # Visual Studio 2008 is missing stdint.h, but you can just download one
    # from the web.
    # http://stackoverflow.com/questions/126279/c99-stdint-h-header-and-ms-visual-studio
    $webclient = New-Object System.Net.WebClient

    $include_dirs = @("C:\Program Files\Microsoft SDKs\Windows\v7.0\Include",
                      "C:\Program Files\Microsoft SDKs\Windows\v7.1\Include",
                      "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include",
                      "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include")

    Foreach ($include_dir in $include_dirs) {
    $urls = @(@("https://raw.githubusercontent.com/chemeris/msinttypes/master/stdint.h", "stdint.h"),
             @("https://raw.githubusercontent.com/chemeris/msinttypes/master/inttypes.h", "inttypes.h"))

    Foreach ($i in $urls) {
        $url = $i[0]
        $filename = $i[1]

        $filepath = "$include_dir\$filename"
        if (Test-Path $filepath) {
            Write-Host $filename "already exists in" $include_dir
            continue
        }

        Write-Host "Downloading remedial " $filename " from" $url "to" $filepath
        $retry_attempts = 2
        for($i=0; $i -lt $retry_attempts; $i++){
            try {
                $webclient.DownloadFile($url, $filepath)
                break
            }
            Catch [Exception]{
                Start-Sleep 1
            }
       }

       if (Test-Path $filepath) {
           Write-Host "File saved at" $filepath
       } else {
           # Retry once to get the error message if any at the last try
           $webclient.DownloadFile($url, $filepath)
       }
    }
    }
}

function main() {
    InstallMissingHeaders
}

main

