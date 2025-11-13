# Mudanças no Cálculo de Kendall's Tau

## Problema Identificado

O fuzzy matching estava **prejudicando** a avaliação:

1. ❌ Reduzia Kendall's Tau de **1.0 → 0.62** (matching incorreto)
2. ❌ Perdia **3 eventos** (encontrava apenas 166/169)
3. ❌ Adicionava **complexidade desnecessária**

## Análise Comparativa

| Estratégia | Kendall's Tau | Eventos Encontrados | Observação |
|------------|---------------|---------------------|------------|
| **Position-based** | **1.0000** | **169/169 (100%)** | Assume ordem cronológica ✅ |
| Fuzzy matching | 0.6183 | 166/169 (98.2%) | Introduz ruído ❌ |
| Cronológico vs Fuzzy | 0.3852 | 167/169 | Pior resultado ❌ |

## Solução Atual

Voltamos a calcular o Kendall's Tau com base na posição relativa das sentenças
encontradas no Golden Sample.

- O resumo gerado e o Golden Sample são segmentados em sentenças.
- Cada sentença do resumo é pareada com a sentença mais semelhante no Golden Sample
    usando similaridade léxica (Jaccard).
- A sequência de posições encontradas é comparada com a ordem esperada via
    `scipy.stats.kendalltau`, produzindo um valor entre -1 e 1.

Essa abordagem volta a refletir o comportamento real do texto gerado, sem assumir
ordem perfeita por construção.

## Resultados Após a Revisão

- ✅ O Kendall's Tau passa a expressar inversões reais de eventos.
- ✅ Mantemos a contagem de eventos compatível com a cobertura efetiva.
- ⚠️ Valores podem ser menores que 1.0 mesmo para fluxos event-by-event se houver
    divergências de conteúdo, expondo problemas de coerência temporal.

## Impacto nas Métricas

| Métrica | Com cálculo real |
|---------|------------------|
| **Kendall's Tau** | Variável conforme o texto gerado |
| **Eventos encontrados** | Depende do casamento por similaridade |
| **ROUGE-1 F1** | Inalterado |
| **ROUGE-2 F1** | Inalterado |
| **METEOR** | Inalterado |
| **BERTScore F1** | Inalterado |

## Justificativa

O Kendall's Tau mede **correlação de ordenação**; assumir valor 1.0 eliminava a
capacidade da métrica de sinalizar regressões. Com o cálculo real, continuamos a
usar a mesma heurística de pareamento de sentenças, mas sem atalhos que travem o
valor mínimo/máximo.
