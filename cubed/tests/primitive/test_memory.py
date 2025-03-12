from cubed.primitive.memory import BufferCopies, MemoryModeller, calculate_projected_mem


def test_calculate_projected_mem():
    projected_mem = calculate_projected_mem(
        reserved_mem=3,
        inputs=[5, 7],
        operation=11,
        output=13,
        buffer_copies=BufferCopies(2, 2),
    )
    assert projected_mem == 3 + 5 + (5 * 2) + 7 + (7 * 2) + 11 + 13 + (13 * 2)


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
