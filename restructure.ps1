# 1. Create directory structure
New-Item -ItemType Directory -Force -Path "src\core"
New-Item -ItemType Directory -Force -Path "src\indexing"
New-Item -ItemType Directory -Force -Path "src\querying"
New-Item -ItemType Directory -Force -Path "src\preprocessing"
New-Item -ItemType Directory -Force -Path "data"
New-Item -ItemType Directory -Force -Path "notebooks"
New-Item -ItemType Directory -Force -Path "tests"

# 2. Create __init__.py files to make them Python packages
$initFiles = @(
    "src\__init__.py",
    "src\core\__init__.py",
    "src\indexing\__init__.py",
    "src\querying\__init__.py",
    "src\preprocessing\__init__.py",
    "tests\__init__.py"
)
foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path $file
}

# 3. Move files to their respective locations
Move-Item -Path "indexing_and_retrieval\index_base.py" -Destination "src\core\"
Move-Item -Path "indexing_and_retrieval\es_indexing.py" -Destination "src\indexing\"
Move-Item -Path "indexing_and_retrieval\self_index*.py" -Destination "src\indexing\"
Move-Item -Path "indexing_and_retrieval\es_querying.py" -Destination "src\querying\"
Move-Item -Path "indexing_and_retrieval\preprocess_data.py" -Destination "src\preprocessing\"

Write-Host "Restructuring complete! Please update your imports to reflect the new package structure."
