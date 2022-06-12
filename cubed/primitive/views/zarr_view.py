from functools import cached_property

from zarr.core import Array


class Translator:
    """Translates Zarr chunk coordinates and chunks for a TranslatedArray."""

    def to_source_chunk_coords(self, target_chunk_coords):
        pass

    def to_target_chunk(self, source_chunk):
        pass


class TranslatedArray(Array):
    """A Zarr Array translated to have a different structure to produce a ZarrArrayView."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def create(cls, array, translator):
        a = TranslatedArray(*array.__getstate__())
        a.translator = translator

        a._shape = translator.shape
        a._chunks = translator.chunks
        a._original_chunks = array.chunks

        return a

    def _chunk_key(self, chunk_coords):
        old_chunk_coords = self.translator.to_source_chunk_coords(chunk_coords)
        return super()._chunk_key(old_chunk_coords)

    def _decode_chunk(self, cdata, start=None, nitems=None, expected_shape=None):
        if expected_shape is not None:
            raise NotImplementedError("TranslatedArray does not support expected_shape")
        old_chunk = super()._decode_chunk(
            cdata, start=start, nitems=nitems, expected_shape=self._original_chunks
        )
        return self.translator.to_target_chunk(old_chunk)


class ZarrArrayView:
    """
    A read-only view of a Zarr Array.

    The view may have a different structure (shape, number of dimensions, chunks) to the underlying array.
    """

    def __init__(self, array, translator):
        self.array = array
        self.translator = translator
        self.shape = translator.shape
        self.dtype = array.dtype
        self.chunks = translator.chunks
        self.is_view = True

    # Don't store as an instance variable, since TranslatedArray is not pickle-able
    @cached_property
    def translated_array(self):
        return TranslatedArray.create(self.array, self.translator)

    def __getitem__(self, selection):
        return self.translated_array.__getitem__(selection)
