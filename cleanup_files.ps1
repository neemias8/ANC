# Script para limpar arquivos desnecessarios do projeto ANC
# Execute com cuidado! Revise antes de executar.

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "LIMPEZA DE ARQUIVOS - Projeto ANC" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$removedCount = 0
$failedCount = 0

function Remove-SafeFile {
    param($filePath)
    
    if (Test-Path $filePath) {
        try {
            Remove-Item $filePath -Force
            Write-Host "[OK] Removido: $filePath" -ForegroundColor Green
            return $true
        } catch {
            Write-Host "[ERRO] Falha ao remover: $filePath" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "[SKIP] Nao encontrado: $filePath" -ForegroundColor Yellow
        return $null
    }
}

Write-Host "1. Removendo scripts de TESTE (17 arquivos)..." -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$testFiles = @(
    "test_adaptive_lengths.py",
    "test_cleaning.py",
    "test_clean_summary.py",
    "test_fixed_numbered_tau.py",
    "test_generation_params.py",
    "test_matching_strategies.py",
    "test_min_new_tokens.py",
    "test_models_long_output.py",
    "test_multi_doc_format.py",
    "test_new_tau.py",
    "test_numbered_tau.py",
    "test_primera_events.py",
    "test_verse_halves.py",
    "verify_tau_calculation.py",
    "debug_missing_events.py",
    "check_chronology_events.py",
    "analyze_conciseness.py"
)

foreach ($file in $testFiles) {
    $result = Remove-SafeFile $file
    if ($result -eq $true) { $removedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host ""
Write-Host "2. Removendo scripts de UTILIDADE/CONVERSAO (6 arquivos)..." -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$utilFiles = @(
    "add_event_numbers.py",
    "fix_numbering_format.py",
    "fix_spacing.py",
    "fix_emojis.py",
    "clean_existing_output.py",
    "save_graph_stats.py"
)

foreach ($file in $utilFiles) {
    $result = Remove-SafeFile $file
    if ($result -eq $true) { $removedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host ""
Write-Host "3. Removendo scripts de DEBUG (2 arquivos)..." -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$debugFiles = @(
    "graph_builder_debug.py",
    "improved_graph_builder.py"
)

foreach ($file in $debugFiles) {
    $result = Remove-SafeFile $file
    if ($result -eq $true) { $removedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host ""
Write-Host "4. Removendo scripts de COMPARACAO ANTIGOS (3 arquivos)..." -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$compareFiles = @(
    "compare_all_methods.py",
    "compare_methods.py",
    "run_evaluate_cleaned.py"
)

foreach ($file in $compareFiles) {
    $result = Remove-SafeFile $file
    if ($result -eq $true) { $removedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host ""
Write-Host "5. Removendo arquivos ORFAOS do src/ (2 arquivos)..." -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$srcFiles = @(
    "src/consolidate_abstractive.py",
    "src/summarize_baseline.py"
)

foreach ($file in $srcFiles) {
    $result = Remove-SafeFile $file
    if ($result -eq $true) { $removedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host ""
Write-Host "6. Removendo documentacao ANTIGA (1 arquivo)..." -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$result = Remove-SafeFile "COMPARISON_LED_vs_PRIMERA_BART_PEGASUS.md"
if ($result -eq $true) { $removedCount++ }
elseif ($result -eq $false) { $failedCount++ }

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "RESUMO DA LIMPEZA" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Arquivos removidos: $removedCount" -ForegroundColor Green
Write-Host "Falhas: $failedCount" -ForegroundColor Red
Write-Host ""
Write-Host "ARQUIVOS ESSENCIAIS PRESERVADOS:" -ForegroundColor Green
Write-Host "  - data/ (todos os XMLs + Golden_Sample.txt)" -ForegroundColor White
Write-Host "  - src/ (main.py, data_loader.py, evaluator.py)" -ForegroundColor White
Write-Host "  - primera_event_consolidation.py" -ForegroundColor White
Write-Host "  - primera_standard_mds.py" -ForegroundColor White
Write-Host "  - test_other_models.py (BART + PEGASUS)" -ForegroundColor White
Write-Host "  - run_taeg.py" -ForegroundColor White
Write-Host "  - compare_primera_simple.py" -ForegroundColor White
Write-Host "  - requirements.txt, pyproject.toml, README.md" -ForegroundColor White
Write-Host "  - KENDALL_TAU_CHANGES.md, DESCOBERTAS_PROYECTO.md" -ForegroundColor White
Write-Host ""
Write-Host "Limpeza concluida!" -ForegroundColor Cyan
