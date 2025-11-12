# 🔬 Descobertas do Projeto: PRIMERA para Consolidação Narrativa

## 📊 Resumo Executivo

Após extensiva experimentação com o modelo PRIMERA para consolidação narrativa de evangelhos, chegamos a conclusões importantes sobre:
1. Como usar corretamente modelos multi-documento (formato `<doc-sep>`)
2. Por que prompts explícitos causam alucinações
3. Parâmetros de geração que funcionam (e os que não funcionam)
4. Comparação entre métodos extrativo (TAEG) vs abstractivos (PRIMERA)

---

## 🎯 Descoberta Principal: PRIMERA Sem Prompts

### O Problema das Alucinações

**Tentativa Inicial (COM prompts):**
```python
prompt = f"Consolidate these {num_gospels} accounts into one narrative. Only use facts from the text. Do not add details:\n\n{combined_text}"
```

**Resultados:**
- ❌ Alucinações severas ("Library of Congress", "Acts of the Apostles")
- ❌ Prompts repetidos no output
- ❌ Meta-texto sobre "como consolidar"
- ❌ Kendall's Tau = 0.50 (ordem cronológica ruim)

### A Solução: Apenas `<doc-sep>`

**Implementação Correta (SEM prompts):**
```python
# Apenas os textos separados por <doc-sep>
input_text = f"{gospel1} <doc-sep> {gospel2} <doc-sep> {gospel3} <doc-sep> {gospel4}"
```

**Resultados:**
- ✅ Zero alucinações
- ✅ Output limpo (sem instruções repetidas)
- ✅ Kendall's Tau = 0.649 (ordem cronológica boa)
- ✅ Comprimento apropriado (~340 chars/evento)

### Por Que Isso Funciona?

**PRIMERA foi treinado para reconhecer `<doc-sep>` como separador de documentos.**

1. **Treinamento Original**: Multi-News, arXiv, PubMed (sem prompts explícitos)
2. **Formato Esperado**: `doc1 <doc-sep> doc2 <doc-sep> doc3`
3. **Tarefa Implícita**: O modelo "sabe" que deve consolidar múltiplos documentos

**Prompts explícitos confundem o modelo:**
- Fora da distribuição de treinamento
- O modelo tenta "seguir instruções" como um chatbot
- Mas não foi treinado para instruction-following
- Resultado: alucinações e repetições

---

## ⚙️ Parâmetros de Geração

### Parâmetros Que Funcionam

```python
# Para PRIMERA-Consolidation (event-based)
max_length_per_event = 256        # Força concisão
min_length_per_event = 10         # Permite eventos curtos
length_penalty = 0.8              # Penaliza textos longos
num_beams = 4                     # Beam search quality
no_repeat_ngram_size = 3          # Evita repetições
do_sample = False                 # Determinístico
repetition_penalty = 1.5          # Penaliza fortemente repetições
```

### Parâmetros Que NÃO Funcionam

```python
# ❌ temperature - NÃO é reconhecido pelo PRIMERA
temperature = 0.3  # IGNORADO com aviso quando do_sample=False
```

**Aviso do modelo:**
```
The following generation flags are not valid and may be ignored: ['temperature']
```

**Explicação**: Com `do_sample=False` (beam search puro), o parâmetro `temperature` não é usado. PRIMERA usa beam search determinístico, não sampling estocástico.

### Evolução dos Parâmetros

| Parâmetro | Tentativa 1 | Tentativa 2 | Versão Final | Efeito |
|-----------|------------|-------------|--------------|--------|
| `max_length_per_event` | 2048 | 512 | **256** | Reduzir alucinações |
| `min_length_per_event` | 50 | 20 | **10** | Permitir eventos curtos |
| `length_penalty` | 1.5 | 1.0 | **0.8** | Penalizar textos longos |
| `temperature` | 0.7 | 0.3 | **removido** | Não funciona com beam search |
| `repetition_penalty` | 1.0 | 1.2 | **1.5** | Evitar repetições |
| `use_event_descriptions` | True | True | **False** | Sem prompts nos eventos |

---

## 📈 Resultados Experimentais

### Comparação Final (10 eventos)

| Métrica | TAEG | PRIMERA-MDS | PRIMERA-Consolidation |
|---------|------|-------------|----------------------|
| **Kendall's Tau** | **1.000** ⭐ | 0.673 | 0.649 |
| **ROUGE-1 F1** | **0.958** ⭐ | 0.017 | 0.075 |
| **ROUGE-2 F1** | **0.938** ⭐ | 0.010 | 0.062 |
| **ROUGE-L F1** | **0.947** ⭐ | 0.012 | 0.057 |
| **BERTScore F1** | **0.995** ⭐ | 0.848 | 0.892 |
| **METEOR** | **0.639** ⭐ | 0.005 | 0.022 |
| **Comprimento** | 79,154 chars | 762 chars | 3,399 chars |
| **Tempo (CPU)** | 35.9s | 56.0s | 63.7s |

### Análise dos Resultados

#### 1. TAEG (Vencedor) ⭐
**Por que venceu:**
- Ordem cronológica perfeita (Tau = 1.000)
- Cobertura completa (ROUGE-L = 0.947)
- Preservação literal do texto original
- Mais rápido (35.9s)

**Limitações:**
- Pode ter quebras de estilo entre sentenças extraídas
- Sem fluência de texto gerado

#### 2. PRIMERA-MDS (Conciso Demais)
**Problema:** Gerou apenas 762 caracteres (1 parágrafo sobre primeiro evento!)
- Ignorou 9 dos 10 eventos
- Comportamento típico de MDS (resumo, não consolidação)
- ROUGE praticamente zero
- **Não adequado para consolidação narrativa**

#### 3. PRIMERA-Consolidation (Promissor, mas insuficiente)
**Pontos Positivos:**
- ✅ Zero alucinações (após descoberta do `<doc-sep>`)
- ✅ Factualmente correto
- ✅ Boa cobertura dos 10 eventos (3,399 chars)

**Pontos Negativos:**
- ⚠️ Ordem cronológica inferior ao TAEG (0.649 vs 1.000)
- ⚠️ ROUGE baixo (texto muito diferente do original)
- ⚠️ Mais lento que TAEG (63.7s vs 35.9s)

---

## 🎓 Lições Aprendidas

### 1. Formato Multi-Documento

**✅ CORRETO:**
```python
# Usar <doc-sep> para separar documentos
input_text = "gospel1_text <doc-sep> gospel2_text <doc-sep> gospel3_text"
```

**❌ ERRADO:**
```python
# Concatenar tudo em um documento único
input_text = "gospel1_text\n\ngospel2_text\n\ngospel3_text"

# Ou adicionar prompts explícitos
input_text = "Consolidate these accounts:\ngospel1_text\ngospel2_text"
```

### 2. Prompts e Instruções

**✅ FAZER:**
- Deixar o modelo fazer o que foi treinado para fazer
- Usar apenas `<doc-sep>` como separador
- Confiar na arquitetura do modelo

**❌ NÃO FAZER:**
- Adicionar prompts em linguagem natural
- Tentar "guiar" o modelo com instruções
- Usar task_prefix detalhado

### 3. Parâmetros de Geração

**✅ FAZER:**
- Usar `do_sample=False` (beam search determinístico)
- Configurar `repetition_penalty` alto (1.5)
- Reduzir `max_length` para forçar concisão
- Usar `length_penalty < 1.0` para penalizar textos longos

**❌ NÃO FAZER:**
- Usar `temperature` com `do_sample=False` (ignorado)
- Forçar textos longos com `length_penalty > 1.0`
- Usar `max_length` muito alto (gera alucinações)

### 4. Escolha do Método

**Para Consolidação de Textos Religiosos/Sagrados:**
- ✅ **Use TAEG** (extrativo)
- Preserva texto original literalmente
- Ordem cronológica perfeita
- Sem risco de alucinações
- Mais rápido

**Para Resumo Conciso:**
- ✅ **Use PRIMERA-MDS**
- Gera parágrafo resumido
- Boa fluência
- Mas perde muitos detalhes

**Para Consolidação com Fluência:**
- ⚠️ **PRIMERA-Consolidation** pode funcionar MAS:
- Requer ajuste fino extensivo
- Ordem cronológica inferior
- Risco de alterar significado
- Não recomendado para textos sagrados

---

## 🔬 Implicações para Pesquisa

### 1. Modelos MDS Pré-treinados Têm Limitações

- PRIMERA foi treinado para **resumir**, não **consolidar**
- "Multi-document" ≠ "Multi-perspective consolidation"
- Viés forte para concisão (não completude)

### 2. Prompting Nem Sempre Funciona

- Nem todos os modelos se beneficiam de prompts
- Modelos pré-treinados funcionam melhor "as designed"
- Adicionar instruções pode causar mais problemas que soluções

### 3. Abordagem Extrativa Tem Vantagens

Para tarefas que exigem:
- Preservação literal do texto
- Ordem cronológica perfeita
- Fidelidade absoluta

**Métodos extrativos (como TAEG) são superiores.**

### 4. Trade-offs Inevitáveis

| Aspecto | Extrativo (TAEG) | Abstractivo (PRIMERA) |
|---------|------------------|----------------------|
| **Fidelidade** | ✅ Perfeita | ⚠️ Pode parafrasear |
| **Fluência** | ⚠️ Pode ter quebras | ✅ Excelente |
| **Ordem Cronológica** | ✅ Perfeita (1.0) | ⚠️ Boa (0.65) |
| **Velocidade** | ✅ Rápido (35s) | ⚠️ Lento (63s) |
| **Risco de Erro** | ✅ Baixo | ⚠️ Alucinações possíveis |

---

## 📝 Recomendações Práticas

### Para Implementar PRIMERA em Projetos Similares

1. **Formato de Entrada**
   ```python
   # Use SEMPRE <doc-sep> entre documentos
   input_text = " <doc-sep> ".join(documents)
   ```

2. **Parâmetros Conservadores**
   ```python
   max_length = 256              # Curto
   length_penalty = 0.8          # Penaliza longo
   repetition_penalty = 1.5      # Evita repetição
   do_sample = False             # Determinístico
   num_beams = 4                 # Quality
   # NÃO usar temperature com beam search!
   ```

3. **Sem Prompts**
   ```python
   # Apenas os documentos, nada mais
   # O modelo sabe o que fazer com <doc-sep>
   ```

4. **Validação de Output**
   ```python
   # Sempre verificar:
   # - Alucinações (fatos não nos documentos)
   # - Repetições de prompt
   # - Meta-texto ("you should...", "this text...")
   ```

### Para Escolher Entre Métodos

**Use TAEG se:**
- Precisa preservar texto literal
- Ordem cronológica é crítica
- Trabalhando com textos sagrados/legais
- Fidelidade > Fluência

**Use PRIMERA-MDS se:**
- Precisa de resumo conciso
- Fluência é prioritária
- Completude não é crítica
- Aceitável perder detalhes

**Use PRIMERA-Consolidation se:**
- Quer experimentar abordagem híbrida
- Disposto a validar manualmente output
- Tem recursos para fine-tuning
- Fluência > Fidelidade literal

---

## 📚 Referências Técnicas

### Sobre PRIMERA

- **Paper**: "PRIMERA: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization" (NAACL 2022)
- **Modelo Base**: Longformer Encoder-Decoder (LED)
- **Contexto**: 16K tokens (4096 × 4 documentos)
- **Treinamento**: Multi-News, arXiv, PubMed
- **Separador Especial**: `<doc-sep>` (token único no vocabulário)

### Sobre Beam Search vs Sampling

- **Beam Search** (`do_sample=False`):
  - Determinístico
  - Explora múltiplas hipóteses simultaneamente
  - Não usa `temperature`
  - Melhor para tarefas factuais

- **Sampling** (`do_sample=True`):
  - Estocástico
  - Usa `temperature` para controlar aleatoriedade
  - Mais criativo
  - Melhor para tarefas criativas

**Para consolidação factual: sempre use Beam Search!**

---

## 🎯 Conclusão Final

### Para Este Projeto (Consolidação de Evangelhos)

**TAEG é a melhor escolha** porque:

1. ✅ Kendall's Tau = 1.000 (ordem perfeita)
2. ✅ ROUGE-L = 0.947 (cobertura quase total)
3. ✅ BERTScore = 0.995 (semelhança máxima)
4. ✅ Mais rápido (35.9s vs 63.7s)
5. ✅ Preserva literalmente o texto sagrado
6. ✅ Sem risco de alucinações

### Contribuição para a Literatura

**Descobertas que avançam o campo:**

1. **Limitações de MDS para Consolidação**: Modelos de resumo multi-documento não são adequados para consolidação narrativa completa

2. **Importância do Formato de Entrada**: `<doc-sep>` é essencial; prompts explícitos causam mais problemas que soluções

3. **Trade-off Fundamental**: Para textos que exigem fidelidade literal (religiosos, legais, científicos), métodos extrativos são superiores aos abstractivos

4. **Validação de Abordagem**: TAEG (extrativo com estrutura temporal explícita) supera estado-da-arte em modelos abstractivos pré-treinados

---

## 📅 Histórico do Projeto

- **Implementação Inicial**: PRIMERA com prompts explícitos → Alucinações severas
- **Ajuste de Parâmetros**: Redução de max_length, temperature → Alucinações continuaram
- **Descoberta Crítica**: Remoção de prompts, uso apenas de `<doc-sep>` → Zero alucinações
- **Otimização Final**: Parâmetros conservadores, sem temperature → Resultados estáveis
- **Comparação 3-Way**: TAEG vs PRIMERA-MDS vs PRIMERA-Consolidation → TAEG vence

**Data**: Novembro de 2025
