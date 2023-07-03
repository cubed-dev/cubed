Search.setIndex({"docnames": ["api", "array-api", "computation", "contributing", "design", "generated/cubed.Array", "generated/cubed.Array.compute", "generated/cubed.Array.rechunk", "generated/cubed.Array.visualize", "generated/cubed.Callback", "generated/cubed.Spec", "generated/cubed.TaskEndEvent", "generated/cubed.apply_gufunc", "generated/cubed.compute", "generated/cubed.from_array", "generated/cubed.from_zarr", "generated/cubed.map_blocks", "generated/cubed.measure_reserved_mem", "generated/cubed.random.random", "generated/cubed.runtime.executors.beam.BeamDagExecutor", "generated/cubed.runtime.executors.lithops.LithopsDagExecutor", "generated/cubed.runtime.executors.modal_async.AsyncModalDagExecutor", "generated/cubed.runtime.executors.python.PythonDagExecutor", "generated/cubed.store", "generated/cubed.to_zarr", "generated/cubed.visualize", "getting-started/demo", "getting-started/index", "getting-started/installation", "getting-started/why-cubed", "index", "operations", "related-projects", "user-guide/executors", "user-guide/index", "user-guide/memory", "user-guide/reliability", "user-guide/storage"], "filenames": ["api.rst", "array-api.md", "computation.md", "contributing.md", "design.md", "generated/cubed.Array.rst", "generated/cubed.Array.compute.rst", "generated/cubed.Array.rechunk.rst", "generated/cubed.Array.visualize.rst", "generated/cubed.Callback.rst", "generated/cubed.Spec.rst", "generated/cubed.TaskEndEvent.rst", "generated/cubed.apply_gufunc.rst", "generated/cubed.compute.rst", "generated/cubed.from_array.rst", "generated/cubed.from_zarr.rst", "generated/cubed.map_blocks.rst", "generated/cubed.measure_reserved_mem.rst", "generated/cubed.random.random.rst", "generated/cubed.runtime.executors.beam.BeamDagExecutor.rst", "generated/cubed.runtime.executors.lithops.LithopsDagExecutor.rst", "generated/cubed.runtime.executors.modal_async.AsyncModalDagExecutor.rst", "generated/cubed.runtime.executors.python.PythonDagExecutor.rst", "generated/cubed.store.rst", "generated/cubed.to_zarr.rst", "generated/cubed.visualize.rst", "getting-started/demo.md", "getting-started/index.md", "getting-started/installation.md", "getting-started/why-cubed.md", "index.md", "operations.md", "related-projects.md", "user-guide/executors.md", "user-guide/index.md", "user-guide/memory.md", "user-guide/reliability.md", "user-guide/storage.md"], "titles": ["API Reference", "Python Array API", "Computation", "Contributing", "Design", "cubed.Array", "cubed.Array.compute", "cubed.Array.rechunk", "cubed.Array.visualize", "cubed.Callback", "cubed.Spec", "cubed.TaskEndEvent", "cubed.apply_gufunc", "cubed.compute", "cubed.from_array", "cubed.from_zarr", "cubed.map_blocks", "cubed.measure_reserved_mem", "cubed.random.random", "cubed.runtime.executors.beam.BeamDagExecutor", "cubed.runtime.executors.lithops.LithopsDagExecutor", "cubed.runtime.executors.modal_async.AsyncModalDagExecutor", "cubed.runtime.executors.python.PythonDagExecutor", "cubed.store", "cubed.to_zarr", "cubed.visualize", "Demo", "Getting Started", "Installation", "Why Cubed?", "Cubed", "Operations", "Related Projects", "Executors", "User Guide", "Memory", "Reliability", "Storage"], "terms": {"A": [0, 2, 4, 29, 35, 36], "cube": [0, 2, 3, 4, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37], "can": [0, 2, 8, 10, 17, 25, 28, 29, 32, 33, 35, 36, 37], "creat": [0, 3, 14, 17, 31, 32, 33, 37], "from_arrai": 0, "from_zarr": 0, "one": [0, 2, 4, 29, 31, 36, 37], "python": [0, 3, 4, 12, 17, 23, 24, 26, 28, 30, 32, 34, 35], "creation": [0, 1], "implement": [1, 4, 17, 23, 26, 29, 30, 31, 32], "array_api": [1, 26], "refer": [1, 12, 30, 33], "its": [1, 4, 7, 31, 35], "specif": [1, 10, 30], "document": 1, "The": [1, 2, 4, 7, 8, 10, 15, 17, 22, 23, 24, 25, 31, 32, 33, 35, 37], "follow": [1, 4, 30, 31, 35, 37], "part": [1, 31, 33], "ar": [1, 2, 3, 4, 12, 28, 29, 31, 32, 33, 35, 36, 37], "categori": 1, "object": [1, 2, 9, 14, 17, 23, 26, 37], "function": [1, 2, 4, 12, 16, 17, 28, 30, 31, 33, 35], "In": [1, 23, 31, 32, 35, 37], "place": 1, "op": 1, "from_dlpack": 1, "index": [1, 4, 31], "boolean": 1, "manipul": 1, "flip": 1, "roll": 1, "search": 1, "nonzero": 1, "set": [1, 2, 10, 12, 17, 28, 35, 36, 37], "unique_al": 1, "unique_count": 1, "unique_invers": 1, "unique_valu": 1, "sort": 1, "argsort": 1, "statist": 1, "std": 1, "var": 1, "accept": 1, "extra": 1, "chunk": [1, 2, 4, 7, 14, 16, 18, 26, 29, 30, 31, 32, 33, 34, 36], "spec": [1, 5, 14, 15, 17, 18, 26, 33, 35, 37], "keyword": [1, 17], "argument": [1, 4, 17], "arang": 1, "start": [1, 26, 29, 30, 33], "stop": 1, "none": [1, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25], "step": 1, "1": [1, 11, 18, 26, 37], "dtype": [1, 2, 4, 16, 31, 35], "devic": 1, "auto": [1, 14], "asarrai": [1, 14, 26], "obj": 1, "copi": 1, "empti": [1, 31], "shape": [1, 4, 7, 31], "empty_lik": 1, "x": [1, 14, 23, 24], "ey": 1, "n_row": 1, "n_col": 1, "k": 1, "0": [1, 10, 18, 28], "full": [1, 28], "fill_valu": 1, "full_lik": 1, "linspac": 1, "num": 1, "endpoint": 1, "true": [1, 6, 8, 13, 25], "ones": [1, 32], "ones_lik": 1, "zero": 1, "zeros_lik": 1, "broadcast_to": 1, "ha": [2, 10, 26, 28, 29, 31, 32, 33, 35, 36], "lazi": [2, 26], "model": [2, 4, 29], "As": [2, 35], "arrai": [2, 9, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 31, 32, 33, 35, 37], "invok": 2, "i": [2, 4, 8, 12, 17, 23, 24, 25, 26, 29, 30, 31, 32, 34, 35, 36, 37], "built": [2, 4], "up": [2, 33, 35, 36, 37], "onli": [2, 29, 31, 32, 33, 36], "when": [2, 10, 17, 32, 33, 35, 36, 37], "explicitli": 2, "trigger": 2, "call": [2, 31, 32, 33, 36], "implicitli": 2, "convert": [2, 32], "an": [2, 3, 4, 7, 8, 10, 14, 15, 17, 19, 20, 21, 24, 25, 29, 31, 34, 35, 36], "numpi": [2, 4, 26, 32], "disk": [2, 4, 8, 25], "zarr": [2, 4, 15, 23, 24, 29, 30, 33, 35, 36, 37], "represent": 2, "direct": 2, "acycl": 2, "graph": [2, 8, 25, 32], "dag": 2, "where": [2, 10, 32], "node": [2, 29], "edg": 2, "express": [2, 4], "primit": [2, 30, 31, 32], "oper": [2, 17, 23, 24, 29, 30, 32, 35, 36, 37], "For": [2, 35, 37], "exampl": [2, 4, 26, 30, 31, 33, 35, 36], "mai": [2, 4, 31, 33, 37], "rechunk": [2, 4, 29, 30, 32], "anoth": [2, 29, 35, 36], "us": [2, 4, 8, 10, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37], "Or": 2, "pair": 2, "ad": 2, "togeth": 2, "blockwis": [2, 4, 30, 32], "both": [2, 32, 36], "have": [2, 32, 33, 35, 36], "requir": [2, 3, 17, 29, 33, 35], "known": [2, 29, 35], "ahead": [2, 35], "time": [2, 29, 31, 35, 36, 37], "each": [2, 31, 35, 36], "run": [2, 10, 17, 22, 23, 24, 26, 28, 29, 32, 33, 35, 36, 37], "task": [2, 4, 10, 11, 12, 17, 22, 32, 33, 35, 36], "output": [2, 8, 12, 24, 25, 31, 35, 36], "need": [2, 4, 12, 31, 35, 37], "size": [2, 18, 29, 31, 34], "natur": [2, 32, 35], "which": [2, 4, 8, 25, 26, 28, 29, 31, 32, 34, 35, 37], "while": [2, 31, 35], "build": [2, 33, 35], "see": [2, 26, 28], "discuss": [2, 33, 35], "travers": 2, "materi": 2, "write": [2, 8, 23, 25, 29, 35, 36, 37], "them": [2, 31, 36, 37], "storag": [2, 10, 15, 24, 29, 30, 34], "detail": [2, 31, 33], "how": [2, 29, 31, 35, 37], "depend": [2, 6, 27, 30, 36], "runtim": [2, 17, 23, 24, 29, 30, 32, 33, 35, 37], "distribut": [2, 29, 32, 36], "choos": 2, "don": [2, 33, 37], "t": [2, 8, 25, 32, 33, 35, 37], "parallel": [2, 29, 32], "effici": [2, 35], "thi": [2, 4, 6, 7, 8, 10, 12, 17, 23, 24, 26, 27, 29, 31, 32, 33, 34, 35, 36, 37], "process": [2, 4, 17, 23, 24, 29, 32, 33, 35], "advantag": [2, 32], "disadvantag": 2, "One": 2, "sinc": [2, 29, 31, 32, 33, 35, 36], "shuffl": [2, 29], "involv": [2, 3, 29], "straightforward": [2, 33], "scale": [2, 29, 36], "veri": [2, 3, 29, 33], "high": [2, 29, 32, 35], "level": [2, 29, 32], "serverless": [2, 29, 32, 33], "environ": [2, 3, 33], "also": [2, 26, 28, 33, 36], "make": [2, 32, 35], "multipl": [2, 4, 12, 13, 16, 25, 30, 31], "engin": [2, 19, 20, 21, 22, 29], "main": 2, "everi": [2, 4, 33], "intermedi": [2, 10, 17, 34], "written": 2, "slow": [2, 36], "howev": [2, 12, 29], "opportun": 2, "optim": [2, 8, 25, 29], "befor": [2, 8, 25, 33, 37], "map": [2, 31], "fusion": 2, "welcom": 3, "pleas": 3, "head": 3, "over": [3, 31, 37], "github": 3, "get": [3, 30, 33, 34, 37], "conda": [3, 27], "name": [3, 5, 8, 25, 32, 37], "3": [3, 26], "8": [3, 26], "activ": 3, "pip": [3, 27, 33], "instal": [3, 17, 27, 30], "r": 3, "txt": [3, 33], "e": 3, "compos": [4, 31], "five": 4, "layer": [4, 32], "from": [4, 14, 15, 16, 29, 30, 31, 33, 35, 36], "bottom": [4, 31], "top": [4, 31], "blue": 4, "block": [4, 16, 31], "green": [4, 31], "red": 4, "other": 4, "project": [4, 28, 30, 34], "like": [4, 14, 23, 26, 29, 31, 32, 35, 37], "beam": [4, 28, 29, 30, 32], "let": 4, "": [4, 21, 29, 30, 31, 32, 35, 37], "go": [4, 29], "through": 4, "back": [4, 33], "mean": [4, 35], "type": [4, 7, 8, 15, 17, 23, 24, 25], "inherit": 4, "attribut": [4, 5, 10, 11, 31], "includ": [4, 8, 10, 25, 32], "underli": 4, "store": [4, 10, 15, 17, 24, 37], "local": [4, 26, 28, 34, 35, 37], "cloud": [4, 17, 26, 29, 32, 34, 35], "well": [4, 32, 35], "unit": [4, 10], "comput": [4, 8, 9, 10, 15, 17, 23, 24, 25, 26, 29, 30, 32, 33, 35, 36, 37], "system": [4, 17, 29, 36, 37], "extern": 4, "It": [4, 17, 31, 32, 33], "algorithm": [4, 31, 35], "deleg": 4, "stateless": [4, 29], "executor": [4, 6, 10, 13, 17, 23, 24, 26, 28, 30, 34, 35, 36, 37], "lithop": [4, 17, 28, 29, 30, 33], "modal": [4, 17, 21, 28, 30, 33], "apach": [4, 19, 28, 29, 30], "There": [4, 35], "two": [4, 31, 33], "appli": [4, 12, 16], "input": [4, 15, 16, 31, 35], "concis": 4, "rule": [4, 35, 37], "chang": [4, 7, 31, 32], "without": [4, 7, 31, 33], "These": 4, "provid": [4, 8, 25, 29, 33, 35], "all": [4, 29, 31, 33], "elemwis": [4, 30], "elementwis": 4, "respect": 4, "broadcast": [4, 31], "map_block": [4, 30], "correspond": [4, 16, 31, 32], "map_direct": [4, 30], "across": 4, "new": [4, 12, 32, 35], "read": [4, 29, 31, 32, 35, 36], "directli": [4, 31], "side": [4, 31], "necessarili": 4, "fashion": [4, 36], "__getitem__": 4, "subset": [4, 26], "along": [4, 31], "more": [4, 12, 31, 32, 33, 35, 36], "ax": [4, 12, 31], "reduct": [4, 30, 35], "reduc": [4, 31], "arg_reduct": [4, 30], "return": [4, 7, 8, 15, 17, 18, 25, 31], "valu": [4, 10, 31, 35], "wa": [4, 32], "chosen": 4, "public": 4, "defin": [4, 32], "few": [4, 29, 36], "extens": [4, 8, 25], "io": [4, 30, 36], "random": [4, 30], "number": [4, 30, 31, 36], "gener": [4, 12, 29, 30, 31, 35], "heavili": [4, 32], "dask": [4, 12, 26, 29, 30], "applic": 4, "class": [5, 9, 10, 11, 19, 20, 21, 22], "zarrai": 5, "plan": [5, 30, 32, 35], "__init__": [5, 9, 10, 11, 19, 20, 21, 22], "method": [5, 9, 10, 11, 19, 20, 21, 22], "callback": [6, 11, 13], "optimize_graph": [6, 8, 13, 25], "resum": [6, 13], "kwarg": [6, 12, 13, 16, 17, 20, 21, 23, 24], "ani": [6, 10, 17, 37], "data": [7, 10, 17, 29, 33, 34, 35], "paramet": [7, 8, 10, 12, 15, 17, 23, 24, 25], "tupl": 7, "desir": 7, "after": [7, 37], "corearrai": [7, 25], "filenam": [8, 25], "format": [8, 25], "produc": [8, 25, 35], "str": [8, 10, 17, 25], "file": [8, 25, 37], "If": [8, 10, 25, 33, 35, 36, 37], "doesn": [8, 25, 35], "svg": [8, 25], "default": [8, 10, 22, 23, 24, 25, 26, 36, 37], "png": [8, 25], "pdf": [8, 25], "dot": [8, 25], "jpeg": [8, 25], "jpg": [8, 25], "option": [8, 10, 15, 17, 23, 24, 25, 27], "bool": [8, 25], "render": [8, 25], "otherwis": [8, 25], "displai": [8, 25], "ipython": [8, 25], "imag": [8, 25], "import": [8, 25, 26, 33, 35, 37], "notebook": [8, 25], "receiv": 9, "event": 9, "dure": [9, 35], "work_dir": [10, 17, 26, 33, 37], "max_mem": 10, "allowed_mem": [10, 26, 33, 35], "reserved_mem": [10, 17, 35], "storage_opt": 10, "resourc": [10, 26], "avail": [10, 26, 35], "specifi": [10, 17, 26, 34, 37], "directori": [10, 17, 37], "path": [10, 15, 17, 24], "fsspec": [10, 17, 37], "url": [10, 17, 37], "int": [10, 17], "deprec": 10, "maximum": [10, 30, 35], "memori": [10, 12, 17, 29, 32, 34], "worker": [10, 29, 33, 35], "byte": [10, 12, 17], "total": [10, 36], "should": [10, 34, 35, 37], "form": [10, 37], "kb": 10, "mb": 10, "gb": 10, "tb": 10, "etc": [10, 30], "been": [10, 26, 29, 32, 33, 36], "reserv": [10, 17, 34], "non": [10, 35], "dict": 10, "pass": [10, 17, 29, 31], "array_nam": 11, "num_task": 11, "task_create_tstamp": 11, "function_start_tstamp": 11, "function_end_tstamp": 11, "task_result_tstamp": 11, "peak_measured_mem_start": 11, "peak_measured_mem_end": 11, "inform": [11, 12, 35], "about": [11, 12, 32], "complet": [11, 36], "func": [12, 16], "signatur": 12, "arg": [12, 16], "axi": [12, 31], "output_dtyp": 12, "output_s": 12, "vector": 12, "ufunc": 12, "similar": [12, 32], "cutdown": 12, "version": [12, 17], "equival": 12, "usag": [12, 29, 30, 35], "current": [12, 23, 36], "limit": 12, "keepdim": 12, "allow_rechunk": 12, "support": [12, 31, 32, 33], "assum": 12, "alloc": [12, 29], "than": [12, 31, 32, 35, 36, 37], "you": [12, 27, 28, 32, 33, 34, 35, 36, 37], "tell": 12, "extra_projected_mem": 12, "amount": [12, 17, 29, 33, 35, 36], "per": 12, "onc": 13, "load": 15, "string": [15, 24], "drop_axi": 16, "new_axi": 16, "measur": [17, 35], "given": [17, 35], "exclud": 17, "vari": 17, "packag": [17, 28, 32], "guid": [17, 27, 30], "work": [17, 29, 30, 31, 33, 35, 36, 37], "trivial": 17, "tini": 17, "peak": 17, "must": [17, 23, 36], "report": [17, 35], "float": 18, "half": 18, "open": 18, "interv": 18, "execut": [19, 20, 21, 22, 30, 35], "async": 21, "api": [21, 26, 30, 31, 32], "sequenti": 22, "loop": 22, "sourc": 23, "target": [23, 29], "save": [23, 24], "note": [23, 24, 31, 33, 36, 37], "eager": [23, 24], "immedi": [23, 24], "collect": 23, "we": [26, 31, 35], "ll": 26, "simpl": [26, 33], "xp": 26, "tmp": 26, "100kb": 26, "2": [26, 28], "4": 26, "5": 26, "6": 26, "7": [26, 28], "9": 26, "standard": [26, 30, 32], "essenti": 26, "convent": 26, "notic": 26, "just": [26, 31], "describ": [26, 33, 36], "b": 26, "c": [26, 28], "add": [26, 31, 37], "evalu": 26, "so": [26, 32, 33, 35, 36, 37], "noth": 26, "yet": [26, 36], "print": 26, "result": 26, "interact": 26, "10": 26, "readm": 26, "servic": [26, 29, 34, 35, 37], "aim": [27, 35], "quickli": [27, 36], "possibl": [27, 29, 35], "why": [27, 30], "demo": [27, 30], "minim": 28, "forg": 28, "m": 28, "mani": [28, 31, 33], "differ": [28, 30, 31, 36], "especi": 28, "diagnost": 28, "To": [28, 36, 37], "list": 28, "optional_depend": 28, "pyproject": 28, "toml": 28, "tqdm": 28, "graphviz": 28, "jinja2": 28, "pydot": 28, "panda": 28, "matplotlib": 28, "seaborn": 28, "gcsf": 28, "aw": [28, 33, 37], "client": 28, "s3f": 28, "test": [28, 33, 36], "runner": [28, 32], "separ": [28, 37], "due": 28, "conflict": 28, "req": 28, "dill": 28, "pytest": 28, "cov": 28, "mock": 28, "manag": [29, 35], "major": 29, "challeng": 29, "design": [29, 30], "framework": 29, "hadoop": 29, "mapreduc": 29, "spark": 29, "purpos": 29, "lead": 29, "widespread": 29, "adopt": 29, "success": 29, "user": [29, 32], "carefulli": 29, "configur": [29, 35, 36], "understand": [29, 32], "break": 29, "program": 29, "abstract": [29, 32], "disproportion": [29, 36], "often": 29, "spent": 29, "tune": 29, "larg": [29, 35, 36], "common": [29, 35], "theme": 29, "here": [29, 31], "most": [29, 33, 34], "interest": 29, "embarrassingli": 29, "between": [29, 30, 31], "lot": [29, 37], "effort": 29, "put": [29, 35], "googl": [29, 32, 33, 37], "dataflow": [29, 32, 33], "lesser": 29, "extent": 29, "undoubtedli": 29, "improv": [29, 35], "perform": [29, 35], "made": 29, "problem": 29, "awai": 29, "approach": [29, 37], "gain": 29, "traction": 29, "last": 29, "year": [29, 32], "formerli": 29, "pywren": 29, "eschew": 29, "central": 29, "do": 29, "everyth": 29, "via": [29, 30], "case": [29, 35, 37], "persist": [29, 37], "n": 29, "dimension": 29, "guarante": [29, 30], "even": 29, "though": 29, "deliber": 29, "avoid": [29, 31], "instead": 29, "bulk": 29, "alwai": 29, "tightli": [29, 35], "control": 29, "therebi": 29, "unpredict": 29, "attempt": [29, 36], "further": [29, 35], "bound": [29, 32, 35], "librari": [30, 32], "integr": [30, 32], "xarrai": 30, "reliabl": [30, 33, 34], "miss": 30, "relat": 30, "previou": [30, 36], "core": [30, 31, 32], "tree": 30, "contribut": 30, "look": 31, "depth": 31, "diagram": 31, "show": [31, 33], "shown": 31, "white": 31, "middl": 31, "orang": 31, "pink": 31, "Not": 31, "repres": 31, "select": [31, 37], "fundament": [31, 32], "simplest": 31, "element": 31, "preserv": 31, "numblock": 31, "singl": [31, 32, 33, 36], "arrow": 31, "order": [31, 35], "clutter": 31, "thei": [31, 32, 35], "match": 31, "too": 31, "squeez": 31, "although": [31, 33], "second": [31, 33], "dimens": 31, "drop": 31, "allow": [31, 34, 36], "regard": 31, "boundari": 31, "No": 31, "turn": 31, "same": [31, 37], "structur": 31, "access": 31, "whatev": [31, 35], "wai": [31, 36], "concat": 31, "sent": 31, "outer": 31, "three": [31, 32, 36], "consult": 31, "page": 31, "repeat": 31, "first": [31, 33, 35], "round": 31, "combin": 31, "would": [31, 35, 37], "until": 31, "similarli": 31, "rather": [31, 37], "flexibl": 32, "sever": 32, "compon": 32, "datafram": 32, "bag": 32, "delai": 32, "decompos": 32, "fine": 32, "grain": 32, "higher": 32, "easier": 32, "visual": 32, "reason": [32, 35], "newer": 32, "wherea": 32, "varieti": [32, 33], "matur": [32, 33], "influenc": 32, "some": [32, 35, 36], "util": [32, 35], "continu": 32, "zappi": 32, "what": 32, "interven": 32, "wasn": 32, "concern": 32, "less": 32, "daunt": 32, "And": 32, "better": 32, "remot": 33, "below": [33, 35], "sometim": 33, "thread": 33, "pythondagexecutor": 33, "intend": 33, "small": 33, "larger": 33, "easiest": 33, "becaus": [33, 35], "handl": 33, "automat": [33, 35, 37], "sign": 33, "free": 33, "account": 33, "300": 33, "gcp": 33, "slightli": 33, "variou": [33, 35], "far": 33, "lambda": 33, "docker": 33, "contain": [33, 35], "1000": 33, "style": 33, "rel": 33, "highest": [33, 35], "overhead": 33, "startup": 33, "minut": 33, "compar": 33, "20": [33, 35], "therefor": 33, "much": [33, 35], "modal_async": 33, "asyncmodaldagexecutor": 33, "s3": [33, 37], "tomwhit": [33, 37], "temp": [33, 37], "2gb": [33, 35], "altern": 33, "abov": 33, "introduc": 34, "concept": 34, "help": [34, 35], "out": [34, 37], "delet": 34, "retri": 34, "timeout": 34, "straggler": 34, "ensur": [35, 37], "never": 35, "exce": 35, "illustr": 35, "diagaram": 35, "your": 35, "machin": 35, "precis": 35, "compress": 35, "conserv": 35, "upper": 35, "projected_mem": 35, "calcul": 35, "greater": 35, "except": [35, 36], "rais": 35, "phase": 35, "check": 35, "confid": 35, "within": 35, "budget": 35, "properli": 35, "measure_reserved_mem": 35, "basi": 35, "baselin": 35, "accur": 35, "estim": 35, "abil": 35, "peak_measured_mem": 35, "actual": 35, "analys": 35, "ran": 35, "room": 35, "thumb": 35, "least": 35, "ten": 35, "decompress": 35, "basic": 35, "four": 35, "itself": 35, "complex": 35, "particular": 35, "sum": 35, "around": 35, "good": 35, "100mb": 35, "factor": 35, "smaller": 35, "plenti": 35, "fault": 36, "toler": 36, "section": 36, "cover": 36, "featur": 36, "fail": 36, "again": 36, "whole": 36, "error": 36, "messag": 36, "take": 36, "longer": 36, "pre": 36, "determin": 36, "consid": 36, "paragraph": 36, "down": 36, "mitig": 36, "specul": 36, "duplic": 36, "launch": 36, "certain": [36, 37], "circumst": 36, "act": 36, "backup": 36, "henc": 36, "bring": 36, "overal": 36, "taken": 36, "origin": 36, "cancel": 36, "expect": 36, "ident": 36, "idempot": 36, "atom": 36, "experiment": 36, "disabl": 36, "filesystem": 37, "By": 37, "temporari": 37, "appropri": 37, "region": 37, "bucket": 37, "doe": 37, "clear": 37, "space": 37, "incur": 37, "unnecessari": 37, "cost": 37, "typic": 37, "remov": 37, "old": 37, "job": 37, "short": 37, "period": 37, "manual": 37, "clean": 37, "tmpdir": 37, "regular": 37, "command": 37, "rm": 37, "On": 37, "conveni": 37, "dedic": 37, "lifecycl": 37, "consol": 37, "click": 37, "tab": 37, "ag": 37, "enter": 37, "dai": 37, "instruct": 37, "sure": 37}, "objects": {"cubed": [[5, 0, 1, "", "Array"], [9, 0, 1, "", "Callback"], [10, 0, 1, "", "Spec"], [11, 0, 1, "", "TaskEndEvent"], [12, 2, 1, "", "apply_gufunc"], [13, 2, 1, "", "compute"], [14, 2, 1, "", "from_array"], [15, 2, 1, "", "from_zarr"], [16, 2, 1, "", "map_blocks"], [17, 2, 1, "", "measure_reserved_mem"], [23, 2, 1, "", "store"], [24, 2, 1, "", "to_zarr"], [25, 2, 1, "", "visualize"]], "cubed.Array": [[5, 1, 1, "", "__init__"], [6, 1, 1, "", "compute"], [7, 1, 1, "", "rechunk"], [8, 1, 1, "", "visualize"]], "cubed.Callback": [[9, 1, 1, "", "__init__"]], "cubed.Spec": [[10, 1, 1, "", "__init__"]], "cubed.TaskEndEvent": [[11, 1, 1, "", "__init__"]], "cubed.array_api": [[1, 2, 1, "", "arange"], [1, 2, 1, "", "asarray"], [1, 2, 1, "", "broadcast_to"], [1, 2, 1, "", "empty"], [1, 2, 1, "", "empty_like"], [1, 2, 1, "", "eye"], [1, 2, 1, "", "full"], [1, 2, 1, "", "full_like"], [1, 2, 1, "", "linspace"], [1, 2, 1, "", "ones"], [1, 2, 1, "", "ones_like"], [1, 2, 1, "", "zeros"], [1, 2, 1, "", "zeros_like"]], "cubed.random": [[18, 2, 1, "", "random"]], "cubed.runtime.executors.beam": [[19, 0, 1, "", "BeamDagExecutor"]], "cubed.runtime.executors.beam.BeamDagExecutor": [[19, 1, 1, "", "__init__"]], "cubed.runtime.executors.lithops": [[20, 0, 1, "", "LithopsDagExecutor"]], "cubed.runtime.executors.lithops.LithopsDagExecutor": [[20, 1, 1, "", "__init__"]], "cubed.runtime.executors.modal_async": [[21, 0, 1, "", "AsyncModalDagExecutor"]], "cubed.runtime.executors.modal_async.AsyncModalDagExecutor": [[21, 1, 1, "", "__init__"]], "cubed.runtime.executors.python": [[22, 0, 1, "", "PythonDagExecutor"]], "cubed.runtime.executors.python.PythonDagExecutor": [[22, 1, 1, "", "__init__"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"]}, "titleterms": {"api": [0, 1, 4], "refer": 0, "arrai": [0, 1, 4, 5, 6, 7, 8, 30], "io": 0, "chunk": [0, 35], "specif": 0, "function": 0, "random": [0, 18], "number": 0, "gener": 0, "runtim": [0, 4, 19, 20, 21, 22], "executor": [0, 19, 20, 21, 22, 33], "python": [1, 22, 33], "miss": 1, "from": 1, "cube": [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30], "differ": 1, "between": 1, "standard": 1, "comput": [2, 6, 13], "plan": 2, "memori": [2, 30, 35], "execut": 2, "contribut": 3, "develop": [3, 30], "design": 4, "storag": [4, 37], "primit": 4, "oper": [4, 31], "core": 4, "rechunk": [7, 31], "visual": [8, 25], "callback": 9, "spec": 10, "taskendev": 11, "apply_gufunc": 12, "from_arrai": 14, "from_zarr": 15, "map_block": [16, 31], "measure_reserved_mem": 17, "beam": 19, "beamdagexecutor": 19, "lithop": 20, "lithopsdagexecutor": 20, "modal_async": 21, "asyncmodaldagexecutor": 21, "pythondagexecutor": 22, "store": 23, "to_zarr": 24, "demo": 26, "get": 27, "start": 27, "instal": 28, "conda": 28, "pip": 28, "option": 28, "depend": [28, 31], "why": 29, "bound": 30, "serverless": 30, "distribut": 30, "n": 30, "dimension": 30, "process": 30, "document": 30, "For": 30, "user": [30, 34], "tree": 31, "elemwis": 31, "map_direct": 31, "blockwis": 31, "reduct": 31, "arg_reduct": 31, "relat": 32, "project": [32, 35], "dask": 32, "xarrai": 32, "previou": 32, "work": 32, "local": 33, "which": 33, "cloud": [33, 37], "servic": 33, "should": 33, "i": 33, "us": 33, "specifi": 33, "an": 33, "guid": 34, "allow": 35, "reserv": 35, "size": 35, "reliabl": 36, "retri": 36, "timeout": 36, "straggler": 36, "delet": 37, "intermedi": 37, "data": 37}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"API Reference": [[0, "api-reference"]], "Array": [[0, "array"]], "IO": [[0, "io"]], "Chunk-specific functions": [[0, "chunk-specific-functions"]], "Random number generation": [[0, "random-number-generation"]], "Runtime": [[0, "runtime"], [4, "runtime"]], "Executors": [[0, "executors"], [33, "executors"]], "Python Array API": [[1, "python-array-api"]], "Missing from Cubed": [[1, "missing-from-cubed"]], "Differences between Cubed and the standard": [[1, "differences-between-cubed-and-the-standard"]], "Computation": [[2, "computation"]], "Plan": [[2, "plan"]], "Memory": [[2, "memory"], [35, "memory"]], "Execution": [[2, "execution"]], "Contributing": [[3, "contributing"]], "Development": [[3, "development"]], "Design": [[4, "design"]], "Storage": [[4, "storage"], [37, "storage"]], "Primitive operations": [[4, "primitive-operations"]], "Core operations": [[4, "core-operations"]], "Array API": [[4, "array-api"]], "cubed.Array": [[5, "cubed-array"]], "cubed.Array.compute": [[6, "cubed-array-compute"]], "cubed.Array.rechunk": [[7, "cubed-array-rechunk"]], "cubed.Array.visualize": [[8, "cubed-array-visualize"]], "cubed.Callback": [[9, "cubed-callback"]], "cubed.Spec": [[10, "cubed-spec"]], "cubed.TaskEndEvent": [[11, "cubed-taskendevent"]], "cubed.apply_gufunc": [[12, "cubed-apply-gufunc"]], "cubed.compute": [[13, "cubed-compute"]], "cubed.from_array": [[14, "cubed-from-array"]], "cubed.from_zarr": [[15, "cubed-from-zarr"]], "cubed.map_blocks": [[16, "cubed-map-blocks"]], "cubed.measure_reserved_mem": [[17, "cubed-measure-reserved-mem"]], "cubed.random.random": [[18, "cubed-random-random"]], "cubed.runtime.executors.beam.BeamDagExecutor": [[19, "cubed-runtime-executors-beam-beamdagexecutor"]], "cubed.runtime.executors.lithops.LithopsDagExecutor": [[20, "cubed-runtime-executors-lithops-lithopsdagexecutor"]], "cubed.runtime.executors.modal_async.AsyncModalDagExecutor": [[21, "cubed-runtime-executors-modal-async-asyncmodaldagexecutor"]], "cubed.runtime.executors.python.PythonDagExecutor": [[22, "cubed-runtime-executors-python-pythondagexecutor"]], "cubed.store": [[23, "cubed-store"]], "cubed.to_zarr": [[24, "cubed-to-zarr"]], "cubed.visualize": [[25, "cubed-visualize"]], "Demo": [[26, "demo"]], "Getting Started": [[27, "getting-started"]], "Installation": [[28, "installation"]], "Conda": [[28, "conda"]], "Pip": [[28, "pip"]], "Optional dependencies": [[28, "optional-dependencies"]], "Why Cubed?": [[29, "why-cubed"]], "Cubed": [[30, "cubed"]], "Bounded-memory serverless distributed N-dimensional array processing": [[30, "bounded-memory-serverless-distributed-n-dimensional-array-processing"]], "Documentation": [[30, "documentation"]], "For users": [[30, null]], "For developers": [[30, null]], "Operations": [[31, "operations"]], "Dependency Tree": [[31, "dependency-tree"]], "elemwise": [[31, "elemwise"]], "map_blocks": [[31, "map-blocks"]], "map_direct": [[31, "map-direct"]], "blockwise": [[31, "blockwise"]], "rechunk": [[31, "rechunk"]], "reduction and arg_reduction": [[31, "reduction-and-arg-reduction"]], "Related Projects": [[32, "related-projects"]], "Dask": [[32, "dask"]], "Xarray": [[32, "xarray"]], "Previous work": [[32, "previous-work"]], "Local Python executor": [[33, "local-python-executor"]], "Which cloud service should I use?": [[33, "which-cloud-service-should-i-use"]], "Specifying an executor": [[33, "specifying-an-executor"]], "User Guide": [[34, "user-guide"]], "Allowed memory": [[35, "allowed-memory"]], "Projected memory": [[35, "projected-memory"]], "Reserved memory": [[35, "reserved-memory"]], "Chunk sizes": [[35, "chunk-sizes"]], "Reliability": [[36, "reliability"]], "Retries": [[36, "retries"]], "Timeouts": [[36, "timeouts"]], "Stragglers": [[36, "stragglers"]], "Cloud storage": [[37, "cloud-storage"]], "Deleting intermediate data": [[37, "deleting-intermediate-data"]]}, "indexentries": {"arange() (in module cubed.array_api)": [[1, "cubed.array_api.arange"]], "asarray() (in module cubed.array_api)": [[1, "cubed.array_api.asarray"]], "broadcast_to() (in module cubed.array_api)": [[1, "cubed.array_api.broadcast_to"]], "empty() (in module cubed.array_api)": [[1, "cubed.array_api.empty"]], "empty_like() (in module cubed.array_api)": [[1, "cubed.array_api.empty_like"]], "eye() (in module cubed.array_api)": [[1, "cubed.array_api.eye"]], "full() (in module cubed.array_api)": [[1, "cubed.array_api.full"]], "full_like() (in module cubed.array_api)": [[1, "cubed.array_api.full_like"]], "linspace() (in module cubed.array_api)": [[1, "cubed.array_api.linspace"]], "ones() (in module cubed.array_api)": [[1, "cubed.array_api.ones"]], "ones_like() (in module cubed.array_api)": [[1, "cubed.array_api.ones_like"]], "zeros() (in module cubed.array_api)": [[1, "cubed.array_api.zeros"]], "zeros_like() (in module cubed.array_api)": [[1, "cubed.array_api.zeros_like"]], "array (class in cubed)": [[5, "cubed.Array"]], "__init__() (cubed.array method)": [[5, "cubed.Array.__init__"]], "compute() (cubed.array method)": [[6, "cubed.Array.compute"]], "rechunk() (cubed.array method)": [[7, "cubed.Array.rechunk"]], "visualize() (cubed.array method)": [[8, "cubed.Array.visualize"]], "callback (class in cubed)": [[9, "cubed.Callback"]], "__init__() (cubed.callback method)": [[9, "cubed.Callback.__init__"]], "spec (class in cubed)": [[10, "cubed.Spec"]], "__init__() (cubed.spec method)": [[10, "cubed.Spec.__init__"]], "taskendevent (class in cubed)": [[11, "cubed.TaskEndEvent"]], "__init__() (cubed.taskendevent method)": [[11, "cubed.TaskEndEvent.__init__"]], "apply_gufunc() (in module cubed)": [[12, "cubed.apply_gufunc"]], "compute() (in module cubed)": [[13, "cubed.compute"]], "from_array() (in module cubed)": [[14, "cubed.from_array"]], "from_zarr() (in module cubed)": [[15, "cubed.from_zarr"]], "map_blocks() (in module cubed)": [[16, "cubed.map_blocks"]], "measure_reserved_mem() (in module cubed)": [[17, "cubed.measure_reserved_mem"]], "random() (in module cubed.random)": [[18, "cubed.random.random"]], "beamdagexecutor (class in cubed.runtime.executors.beam)": [[19, "cubed.runtime.executors.beam.BeamDagExecutor"]], "__init__() (cubed.runtime.executors.beam.beamdagexecutor method)": [[19, "cubed.runtime.executors.beam.BeamDagExecutor.__init__"]], "lithopsdagexecutor (class in cubed.runtime.executors.lithops)": [[20, "cubed.runtime.executors.lithops.LithopsDagExecutor"]], "__init__() (cubed.runtime.executors.lithops.lithopsdagexecutor method)": [[20, "cubed.runtime.executors.lithops.LithopsDagExecutor.__init__"]], "asyncmodaldagexecutor (class in cubed.runtime.executors.modal_async)": [[21, "cubed.runtime.executors.modal_async.AsyncModalDagExecutor"]], "__init__() (cubed.runtime.executors.modal_async.asyncmodaldagexecutor method)": [[21, "cubed.runtime.executors.modal_async.AsyncModalDagExecutor.__init__"]], "pythondagexecutor (class in cubed.runtime.executors.python)": [[22, "cubed.runtime.executors.python.PythonDagExecutor"]], "__init__() (cubed.runtime.executors.python.pythondagexecutor method)": [[22, "cubed.runtime.executors.python.PythonDagExecutor.__init__"]], "store() (in module cubed)": [[23, "cubed.store"]], "to_zarr() (in module cubed)": [[24, "cubed.to_zarr"]], "visualize() (in module cubed)": [[25, "cubed.visualize"]]}})