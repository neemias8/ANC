"""
Teste para verificar se PRIMERA está usando múltiplos documentos corretamente.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocess import PrimeraPreprocessor


def test_mds_input_format():
    """Verifica se prepare_mds_input usa <doc-sep>."""
    print("\n" + "="*70)
    print("TESTE 1: MDS Input Format")
    print("="*70)
    
    preprocessor = PrimeraPreprocessor()
    mds_input = preprocessor.prepare_mds_input()
    
    # Check for <doc-sep> token
    has_separator = "<doc-sep>" in mds_input
    num_docs = mds_input.count("<doc-sep>") + 1 if has_separator else 1
    
    print(f"\n✓ MDS Input Length: {len(mds_input):,} characters")
    print(f"✓ Contains <doc-sep>: {has_separator}")
    print(f"✓ Number of documents: {num_docs}")
    
    if has_separator and num_docs == 4:
        print("\n✅ PASSOU: MDS input usa 4 documentos separados com <doc-sep>")
        return True
    else:
        print("\n❌ FALHOU: MDS input NÃO usa múltiplos documentos corretamente")
        print(f"   Esperado: 4 documentos com <doc-sep>")
        print(f"   Encontrado: {num_docs} documento(s)")
        return False


def test_event_based_input_format():
    """Verifica se prepare_event_based_inputs usa <doc-sep>."""
    print("\n" + "="*70)
    print("TESTE 2: Event-Based Input Format")
    print("="*70)
    
    preprocessor = PrimeraPreprocessor()
    event_inputs = preprocessor.prepare_event_based_inputs()
    
    # Find events with multiple gospels
    multi_gospel_events = [e for e in event_inputs if e['num_gospels'] > 1]
    
    print(f"\n✓ Total events: {len(event_inputs)}")
    print(f"✓ Multi-gospel events: {len(multi_gospel_events)}")
    
    # Check first multi-gospel event
    if multi_gospel_events:
        event = multi_gospel_events[0]
        has_separator = "<doc-sep>" in event['combined_text']
        num_docs = event['combined_text'].count("<doc-sep>") + 1 if has_separator else 1
        
        print(f"\n📋 Exemplo: Event {event['event_id']}")
        print(f"   Description: {event['description'][:60]}...")
        print(f"   Num gospels: {event['num_gospels']}")
        print(f"   Contains <doc-sep>: {has_separator}")
        print(f"   Detected documents: {num_docs}")
        print(f"   Text preview: {event['combined_text'][:100]}...")
        
        if has_separator and num_docs == event['num_gospels']:
            print(f"\n✅ PASSOU: Evento usa {num_docs} documentos separados com <doc-sep>")
            return True
        else:
            print(f"\n❌ FALHOU: Evento NÃO usa múltiplos documentos corretamente")
            print(f"   Esperado: {event['num_gospels']} documentos com <doc-sep>")
            print(f"   Encontrado: {num_docs} documento(s)")
            return False
    else:
        print("\n⚠️  AVISO: Nenhum evento multi-gospel encontrado para testar")
        return True


def test_no_gospel_labels_in_text():
    """Verifica se removemos os labels [Matthew], [Mark], etc."""
    print("\n" + "="*70)
    print("TESTE 3: No Gospel Labels in Multi-Doc Format")
    print("="*70)
    
    preprocessor = PrimeraPreprocessor()
    event_inputs = preprocessor.prepare_event_based_inputs()
    
    # Check if any event still has gospel labels like [Matthew]
    events_with_labels = []
    for event in event_inputs:
        text = event['combined_text']
        if '[Matthew]' in text or '[Mark]' in text or '[Luke]' in text or '[John]' in text:
            events_with_labels.append(event['event_id'])
    
    if events_with_labels:
        print(f"\n❌ FALHOU: {len(events_with_labels)} eventos ainda têm labels [Gospel]")
        print(f"   Exemplos: {events_with_labels[:5]}")
        print(f"   Labels devem ser removidos ao usar <doc-sep>")
        return False
    else:
        print(f"\n✅ PASSOU: Nenhum evento tem labels [Gospel] no texto")
        print(f"   Documentos separados apenas por <doc-sep>")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 TESTES DE FORMATO MULTI-DOCUMENTO DO PRIMERA")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("MDS Format", test_mds_input_format()))
    results.append(("Event Format", test_event_based_input_format()))
    results.append(("No Labels", test_no_gospel_labels_in_text()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESUMO DOS TESTES")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n🎯 Total: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n✅ TODOS OS TESTES PASSARAM!")
        print("   PRIMERA está usando múltiplos documentos corretamente.")
        return 0
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")
        print("   Verifique a implementação do formato multi-documento.")
        return 1


if __name__ == "__main__":
    exit(main())
