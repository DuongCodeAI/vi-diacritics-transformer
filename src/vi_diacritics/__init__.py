__version__ = "0.1.0"


def __getattr__(name):
    # import muộn để `import vi_diacritics` không bắt buộc có onnxruntime
    if name == "Restorer":
        from .restorer import Restorer

        return Restorer
    raise AttributeError(name)
