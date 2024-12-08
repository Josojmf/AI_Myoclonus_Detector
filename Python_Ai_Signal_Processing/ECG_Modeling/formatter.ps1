# Define input and output file paths
$inputFile = "ECG-filtered_signal.csv"
$outputFile = "ECG-filtered_signal.csv"

# Read the file line by line
$fileContent = Get-Content -Path $inputFile

# Open the output file for writing
$output = @()

# Process the file
for ($i = 0; $i -lt $fileContent.Count; $i++) {
    # Process the header (first line)
    if ($i -eq 0) {
        $output += $fileContent[$i] # Keep the first line as-is
    } else {
        # Split the line by commas, remove the last column, and rejoin
        $columns = $fileContent[$i] -split ","
        $modifiedLine = ($columns[0..($columns.Count - 2)] -join ",")
        $output += $modifiedLine
    }
}

# Save the output to a new file
$output | Set-Content -Path $outputFile

Write-Host "File processed successfully. Output saved to $outputFile"
