from cubed.primitive.types import MemoryModeller


def test_memory_modeller():
    modeller = MemoryModeller()
    assert modeller.current_mem == 0
    assert modeller.peak_mem == 0

    modeller.allocate(100)
    assert modeller.current_mem == 100
    assert modeller.peak_mem == 100

    modeller.free(50)
    assert modeller.current_mem == 50
    assert modeller.peak_mem == 100
