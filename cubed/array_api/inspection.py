from cubed.backend_array_api import namespace as nxp


class __array_namespace_info__:
    # capabilities are determined by Cubed, not the backend array API
    def capabilities(self):
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
        }

    # devices and dtypes are determined by the backend array API

    def default_device(self):
        return nxp.__array_namespace_info__().default_device()

    def default_dtypes(self, *, device=None):
        return nxp.__array_namespace_info__().default_dtypes(device=device)

    def devices(self):
        return nxp.__array_namespace_info__().devices()

    def dtypes(self, *, device=None, kind=None):
        return nxp.__array_namespace_info__().dtypes(device=device, kind=kind)
