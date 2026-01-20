from cubed.backend_array_api import namespace as nxp


class __array_namespace_info__:
    def capabilities(self):
        cap = {
            "boolean indexing": False,  # not supported in Cubed (#73)
            "data-dependent shapes": False,  # not supported in Cubed
        }
        nxp_cap = nxp.__array_namespace_info__().capabilities()
        if "max dimensions" in nxp_cap:
            cap["max dimensions"] = nxp_cap["max dimensions"]
        return cap

    # devices and dtypes are determined by the backend array API

    def default_device(self):
        return nxp.__array_namespace_info__().default_device()

    def default_dtypes(self, *, device=None):
        return nxp.__array_namespace_info__().default_dtypes(device=device)

    def devices(self):
        return nxp.__array_namespace_info__().devices()

    def dtypes(self, *, device=None, kind=None):
        return nxp.__array_namespace_info__().dtypes(device=device, kind=kind)
