Search.setIndex({"docnames": ["api", "array_api", "computation", "contributing", "design", "generated/cubed.Array", "generated/cubed.Array.compute", "generated/cubed.Array.rechunk", "generated/cubed.Array.visualize", "generated/cubed.Callback", "generated/cubed.Spec", "generated/cubed.TaskEndEvent", "generated/cubed.apply_gufunc", "generated/cubed.compute", "generated/cubed.from_array", "generated/cubed.from_zarr", "generated/cubed.map_blocks", "generated/cubed.measure_reserved_memory", "generated/cubed.random.random", "generated/cubed.runtime.executors.beam.BeamDagExecutor", "generated/cubed.runtime.executors.lithops.LithopsDagExecutor", "generated/cubed.runtime.executors.modal_async.AsyncModalDagExecutor", "generated/cubed.runtime.executors.python.PythonDagExecutor", "generated/cubed.store", "generated/cubed.to_zarr", "generated/cubed.visualize", "getting_started", "index", "operations", "related_projects"], "filenames": ["api.rst", "array_api.md", "computation.md", "contributing.md", "design.md", "generated/cubed.Array.rst", "generated/cubed.Array.compute.rst", "generated/cubed.Array.rechunk.rst", "generated/cubed.Array.visualize.rst", "generated/cubed.Callback.rst", "generated/cubed.Spec.rst", "generated/cubed.TaskEndEvent.rst", "generated/cubed.apply_gufunc.rst", "generated/cubed.compute.rst", "generated/cubed.from_array.rst", "generated/cubed.from_zarr.rst", "generated/cubed.map_blocks.rst", "generated/cubed.measure_reserved_memory.rst", "generated/cubed.random.random.rst", "generated/cubed.runtime.executors.beam.BeamDagExecutor.rst", "generated/cubed.runtime.executors.lithops.LithopsDagExecutor.rst", "generated/cubed.runtime.executors.modal_async.AsyncModalDagExecutor.rst", "generated/cubed.runtime.executors.python.PythonDagExecutor.rst", "generated/cubed.store.rst", "generated/cubed.to_zarr.rst", "generated/cubed.visualize.rst", "getting_started.md", "index.md", "operations.md", "related_projects.md"], "titles": ["API Reference", "Python Array API", "Computation", "Contributing", "Design", "cubed.Array", "cubed.Array.compute", "cubed.Array.rechunk", "cubed.Array.visualize", "cubed.Callback", "cubed.Spec", "cubed.TaskEndEvent", "cubed.apply_gufunc", "cubed.compute", "cubed.from_array", "cubed.from_zarr", "cubed.map_blocks", "cubed.measure_reserved_memory", "cubed.random.random", "cubed.runtime.executors.beam.BeamDagExecutor", "cubed.runtime.executors.lithops.LithopsDagExecutor", "cubed.runtime.executors.modal_async.AsyncModalDagExecutor", "cubed.runtime.executors.python.PythonDagExecutor", "cubed.store", "cubed.to_zarr", "cubed.visualize", "Getting Started", "Cubed", "Operations", "Related Projects"], "terms": {"A": [0, 2, 4, 27], "cube": [0, 2, 3, 4, 26, 28, 29], "can": [0, 2, 8, 17, 25, 26, 27], "creat": [0, 3, 14, 17, 28, 29], "from_arrai": 0, "from_zarr": 0, "one": [0, 2, 4, 27, 28], "python": [0, 2, 3, 4, 12, 17, 23, 24, 26, 27, 29], "creation": [0, 1], "implement": [1, 4, 17, 23, 27, 28, 29], "array_api": [1, 26], "refer": [1, 12, 27], "its": [1, 2, 4, 7, 28], "specif": [1, 10, 27], "document": 1, "The": [1, 2, 4, 7, 8, 15, 17, 22, 23, 24, 25, 28, 29], "follow": [1, 4, 27, 28], "part": [1, 28], "ar": [1, 2, 3, 4, 12, 26, 27, 28, 29], "categori": 1, "object": [1, 2, 9, 14, 17, 23], "function": [1, 2, 4, 12, 16, 26, 27, 28], "In": [1, 23, 28, 29], "place": 1, "op": 1, "from_dlpack": 1, "index": [1, 4, 28], "boolean": 1, "manipul": 1, "flip": 1, "roll": 1, "search": 1, "nonzero": 1, "set": [1, 2, 12, 17, 26], "unique_al": 1, "unique_count": 1, "unique_invers": 1, "unique_valu": 1, "sort": 1, "argsort": 1, "statist": 1, "std": 1, "var": 1, "accept": 1, "extra": 1, "chunk": [1, 2, 4, 7, 14, 16, 18, 26, 27, 28, 29], "spec": [1, 2, 5, 14, 15, 17, 18, 26], "keyword": 1, "argument": [1, 4], "arang": 1, "start": [1, 2, 27], "stop": 1, "none": [1, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 23, 24, 25], "step": 1, "1": [1, 11, 18, 26], "dtype": [1, 2, 4, 16, 28], "devic": 1, "auto": [1, 14], "asarrai": [1, 14], "obj": 1, "copi": 1, "empti": [1, 28], "shape": [1, 4, 7, 28], "empty_lik": 1, "x": [1, 14, 23, 24], "ey": 1, "n_row": 1, "n_col": 1, "k": 1, "0": [1, 10, 18, 26], "full": [1, 26], "fill_valu": 1, "full_lik": 1, "linspac": 1, "num": 1, "endpoint": 1, "true": [1, 6, 8, 13, 25], "ones": [1, 29], "ones_lik": 1, "zero": 1, "zeros_lik": 1, "broadcast_to": 1, "ha": [2, 26, 27, 28, 29], "lazi": 2, "model": [2, 4, 27], "As": 2, "arrai": [2, 9, 12, 13, 14, 15, 16, 23, 24, 25, 26, 28, 29], "invok": 2, "i": [2, 4, 8, 12, 17, 23, 24, 25, 27, 28, 29], "built": [2, 4], "up": 2, "onli": [2, 27, 28, 29], "when": [2, 17, 29], "explicitli": 2, "trigger": 2, "call": [2, 28, 29], "implicitli": 2, "convert": [2, 29], "an": [2, 3, 4, 7, 8, 14, 15, 17, 19, 20, 21, 24, 25, 27, 28], "numpi": [2, 4, 29], "disk": [2, 4, 8, 25], "zarr": [2, 4, 15, 23, 24, 27], "represent": 2, "direct": 2, "acycl": 2, "graph": [2, 8, 25, 29], "dag": 2, "where": [2, 29], "node": [2, 27], "edg": 2, "express": [2, 4], "primit": [2, 27, 28, 29], "oper": [2, 17, 23, 24, 27, 29], "For": 2, "exampl": [2, 4, 27, 28], "mai": [2, 4, 28], "rechunk": [2, 4, 27, 29], "anoth": [2, 27], "us": [2, 4, 8, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "Or": 2, "pair": 2, "ad": 2, "togeth": 2, "blockwis": [2, 4, 27, 29], "both": [2, 29], "have": [2, 29], "requir": [2, 3, 27], "known": [2, 27], "ahead": 2, "time": [2, 27, 28], "each": [2, 28], "run": [2, 10, 17, 22, 23, 24, 26, 27, 29], "task": [2, 4, 11, 12, 17, 22, 29], "output": [2, 8, 12, 24, 25, 28], "need": [2, 4, 12, 28], "size": [2, 18, 27, 28], "natur": [2, 29], "which": [2, 4, 8, 25, 26, 27, 28, 29], "while": [2, 28], "build": 2, "possibl": [2, 27], "gener": [2, 4, 12, 27, 28], "precis": 2, "amount": [2, 12, 17, 27], "sinc": [2, 27, 28, 29], "how": [2, 27, 28], "well": [2, 4, 29], "compress": 2, "put": [2, 27], "conserv": 2, "upper": 2, "bound": 2, "usag": [2, 12, 27], "thi": [2, 4, 6, 7, 8, 12, 17, 23, 24, 27, 28, 29], "projected_mem": 2, "calcul": 2, "automat": 2, "maximum": [2, 27], "allow": [2, 28], "must": [2, 17, 23], "specifi": 2, "user": [2, 29], "done": 2, "allowed_mem": [2, 10, 26], "paramet": [2, 7, 8, 12, 15, 17, 23, 24, 25], "If": [2, 8, 25], "greater": 2, "than": [2, 12, 28, 29], "valu": [2, 4, 28], "except": 2, "rais": 2, "dure": [2, 9], "phase": 2, "check": 2, "mean": [2, 4], "high": [2, 27, 29], "confid": 2, "within": 2, "budget": 2, "It": [2, 4, 17, 28, 29], "also": [2, 26], "good": 2, "idea": 2, "reserved_mem": [2, 10, 17], "reserv": [2, 17], "worker": [2, 27], "non": 2, "data": [2, 7, 17, 27], "": [2, 4, 21, 27, 28, 29], "whatev": [2, 28], "process": [2, 4, 17, 23, 24], "estim": 2, "measure_reserved_memori": 2, "baselin": 2, "order": [2, 28], "more": [2, 4, 12, 26, 28, 29], "accur": 2, "actual": 2, "peak": [2, 17], "measur": [2, 17], "analys": 2, "see": [2, 26], "project": [2, 4, 26, 27], "match": [2, 28], "peak_measured_mem": 2, "historycallback": 2, "detail": [2, 28], "travers": 2, "materi": 2, "write": [2, 8, 23, 25, 27], "them": [2, 28], "storag": [2, 15, 24, 27], "depend": [2, 6, 26, 27, 29], "distribut": [2, 29], "choos": 2, "don": 2, "t": [2, 8, 25, 29], "parallel": [2, 27, 29], "effici": 2, "advantag": [2, 29], "disadvantag": 2, "One": 2, "shuffl": [2, 27], "involv": [2, 3, 27], "straightforward": 2, "scale": [2, 27], "veri": [2, 3, 27], "level": [2, 27, 29], "serverless": 2, "environ": [2, 3], "make": [2, 29], "multipl": [2, 4, 12, 13, 16, 25, 27, 28], "engin": [2, 19, 20, 21, 22, 27], "main": 2, "everi": [2, 4], "intermedi": 2, "written": 2, "slow": 2, "howev": [2, 12, 27], "opportun": 2, "optim": [2, 8, 25, 27], "befor": [2, 8, 25], "map": [2, 28], "fusion": 2, "failur": 2, "handl": 2, "fail": 2, "io": [2, 4, 27], "read": [2, 4, 27, 28], "retri": 2, "total": 2, "three": [2, 28, 29], "attempt": [2, 27], "resum": 2, "from": [2, 4, 14, 15, 16, 27, 28], "checkpoint": 2, "persist": [2, 27], "without": [2, 4, 7, 28], "scratch": 2, "To": [2, 26], "do": [2, 27], "should": 2, "store": [2, 4, 15, 24], "dill": [2, 26], "so": [2, 29], "reload": 2, "new": [2, 4, 12, 29], "finish": 2, "straggler": 2, "mitig": 2, "few": [2, 4, 27], "disproportion": [2, 27], "down": 2, "whole": 2, "specul": 2, "duplic": 2, "launch": 2, "certain": 2, "circumst": 2, "act": 2, "backup": 2, "complet": [2, 11], "quickli": 2, "henc": 2, "bring": 2, "overal": 2, "taken": 2, "welcom": 3, "pleas": 3, "head": 3, "over": [3, 28], "github": 3, "get": [3, 27], "conda": 3, "name": [3, 5, 8, 25, 29], "3": 3, "8": 3, "activ": 3, "pip": 3, "instal": [3, 17, 27], "r": 3, "txt": 3, "e": 3, "compos": [4, 28], "five": 4, "layer": [4, 29], "bottom": [4, 28], "top": [4, 28], "blue": 4, "block": [4, 16, 28], "green": [4, 28], "red": 4, "other": 4, "like": [4, 14, 23, 27, 28, 29], "beam": [4, 26, 27, 29], "let": 4, "go": [4, 27], "through": 4, "back": 4, "type": [4, 7, 8, 15, 17, 23, 24, 25], "inherit": 4, "attribut": [4, 5, 10, 11, 28], "includ": [4, 8, 25, 29], "underli": 4, "local": [4, 26], "cloud": [4, 26, 27, 29], "unit": 4, "comput": [4, 8, 9, 10, 15, 17, 23, 24, 25, 26, 27, 29], "system": [4, 17, 27], "extern": 4, "deleg": 4, "stateless": [4, 27], "executor": [4, 6, 10, 13, 17, 23, 24, 26, 27], "lithop": [4, 17, 26, 27], "modal": [4, 17, 21, 26, 27], "dask": [4, 12, 27], "prefect": 4, "There": 4, "two": [4, 28], "appli": [4, 12, 16], "input": [4, 15, 16, 28], "concis": 4, "rule": 4, "chang": [4, 7, 28, 29], "These": 4, "provid": [4, 8, 25, 27], "all": [4, 27, 28], "elemwis": [4, 27], "elementwis": 4, "respect": 4, "broadcast": [4, 28], "map_block": [4, 27], "correspond": [4, 16, 28, 29], "map_direct": [4, 27], "across": 4, "directli": [4, 28], "side": [4, 28], "necessarili": 4, "fashion": 4, "__getitem__": 4, "subset": 4, "along": [4, 28], "ax": [4, 12, 28], "reduct": [4, 27], "reduc": [4, 28], "arg_reduct": [4, 27], "return": [4, 7, 8, 15, 17, 18, 25, 28], "wa": [4, 29], "chosen": 4, "public": 4, "defin": [4, 29], "extens": [4, 8, 25], "random": [4, 26, 27], "number": [4, 27, 28], "heavili": [4, 29], "applic": 4, "class": [5, 9, 10, 11, 19, 20, 21, 22], "zarrai": 5, "plan": [5, 27, 29], "__init__": [5, 9, 10, 11, 19, 20, 21, 22], "method": [5, 9, 10, 11, 19, 20, 21, 22], "callback": [6, 11, 13], "optimize_graph": [6, 8, 13, 25], "kwarg": [6, 12, 13, 16, 20, 23, 24], "ani": [6, 17], "tupl": 7, "desir": 7, "after": 7, "corearrai": [7, 25], "filenam": [8, 25], "format": [8, 25], "produc": [8, 25], "str": [8, 25], "file": [8, 25], "doesn": [8, 25], "svg": [8, 25], "default": [8, 22, 23, 24, 25], "png": [8, 25], "pdf": [8, 25], "dot": [8, 25], "jpeg": [8, 25], "jpg": [8, 25], "option": [8, 15, 23, 24, 25, 26], "bool": [8, 25], "render": [8, 25], "otherwis": [8, 25], "displai": [8, 25], "ipython": [8, 25], "imag": [8, 25], "import": [8, 25, 26], "notebook": [8, 25, 26], "receiv": 9, "event": 9, "work_dir": [10, 26], "max_mem": 10, "storage_opt": 10, "resourc": 10, "avail": 10, "array_nam": 11, "num_task": 11, "task_create_tstamp": 11, "function_start_tstamp": 11, "function_end_tstamp": 11, "task_result_tstamp": 11, "peak_measured_mem_start": 11, "peak_measured_mem_end": 11, "inform": [11, 12], "about": [11, 12, 26, 29], "func": [12, 16], "signatur": 12, "arg": [12, 16], "axi": [12, 28], "output_dtyp": 12, "output_s": 12, "vector": 12, "ufunc": 12, "similar": [12, 29], "cutdown": 12, "version": [12, 17], "equival": 12, "current": [12, 23], "limit": 12, "keepdim": 12, "allow_rechunk": 12, "support": [12, 28, 29], "assum": 12, "alloc": [12, 27], "memori": [12, 17, 29], "you": [12, 26], "tell": 12, "extra_projected_mem": 12, "byte": [12, 17], "per": 12, "onc": 13, "load": 15, "string": [15, 24], "path": [15, 24], "drop_axi": 16, "new_axi": 16, "given": 17, "runtim": [17, 23, 24, 27, 29], "exclud": 17, "vari": 17, "packag": [17, 26], "guid": 17, "work": [17, 27, 28], "trivial": 17, "tini": 17, "report": 17, "int": 17, "float": 18, "half": 18, "open": 18, "interv": 18, "execut": [19, 20, 21, 22, 27], "apach": [19, 26, 27], "async": 21, "api": [21, 27, 28, 29], "sequenti": 22, "loop": 22, "sourc": 23, "target": [23, 27], "save": [23, 24], "note": [23, 24, 28], "eager": [23, 24], "immedi": [23, 24], "collect": 23, "minim": 26, "c": 26, "forg": 26, "m": 26, "mani": [26, 28], "differ": [26, 27, 28], "especi": 26, "diagnost": 26, "list": 26, "optional_depend": 26, "pyproject": 26, "toml": 26, "tqdm": 26, "graphviz": 26, "pydot": 26, "panda": 26, "gcsf": 26, "aw": 26, "2": 26, "7": 26, "client": 26, "s3f": 26, "test": 26, "runner": [26, 29], "separ": 26, "due": 26, "conflict": 26, "req": 26, "pytest": 26, "cov": 26, "mock": 26, "simpl": 26, "xp": 26, "tmp": 26, "100_000": 26, "4": 26, "b": 26, "matmul": 26, "22171031": 26, "93644194": 26, "83459119": 26, "8087655": 26, "3540541": 26, "13054495": 26, "24504742": 26, "05022751": 26, "98211893": 26, "62740696": 26, "21686602": 26, "26402294": 26, "58566331": 26, "33010476": 26, "3994953": 26, "29258764": 26, "demo": 26, "readm": 26, "servic": [26, 27], "librari": [27, 29], "standard": [27, 29], "guarante": 27, "etc": 27, "manag": 27, "major": 27, "challeng": 27, "design": 27, "framework": 27, "hadoop": 27, "mapreduc": 27, "spark": 27, "purpos": 27, "lead": 27, "widespread": 27, "adopt": 27, "success": 27, "carefulli": 27, "configur": 27, "understand": [27, 29], "break": 27, "program": 27, "abstract": [27, 29], "often": 27, "spent": 27, "tune": 27, "larg": 27, "common": 27, "theme": 27, "here": [27, 28], "most": 27, "interest": 27, "embarrassingli": 27, "between": [27, 28], "lot": 27, "effort": 27, "been": [27, 29], "googl": [27, 29], "dataflow": [27, 29], "lesser": 27, "extent": 27, "undoubtedli": 27, "improv": 27, "perform": 27, "made": 27, "problem": 27, "awai": 27, "approach": 27, "gain": 27, "traction": 27, "last": 27, "year": [27, 29], "formerli": 27, "pywren": 27, "eschew": 27, "central": 27, "everyth": 27, "via": 27, "case": 27, "even": 27, "though": 27, "deliber": 27, "avoid": [27, 28], "pass": [27, 28], "instead": 27, "bulk": 27, "alwai": 27, "tightli": 27, "control": 27, "therebi": 27, "unpredict": 27, "further": 27, "miss": 27, "relat": 27, "previou": 27, "core": [27, 28, 29], "tree": 27, "featur": 27, "contribut": 27, "we": 28, "look": 28, "depth": 28, "diagram": 28, "show": 28, "shown": 28, "white": 28, "middl": 28, "orang": 28, "pink": 28, "Not": 28, "just": 28, "repres": 28, "select": 28, "fundament": [28, 29], "simplest": 28, "element": 28, "preserv": 28, "numblock": 28, "singl": [28, 29], "add": 28, "arrow": 28, "clutter": 28, "thei": [28, 29], "too": 28, "squeez": 28, "although": 28, "second": 28, "dimens": 28, "drop": 28, "regard": 28, "boundari": 28, "No": 28, "turn": 28, "same": 28, "structur": 28, "access": 28, "wai": 28, "concat": 28, "sent": 28, "outer": 28, "consult": 28, "page": 28, "algorithm": 28, "repeat": 28, "first": 28, "round": 28, "combin": 28, "would": 28, "until": 28, "similarli": 28, "rather": 28, "flexibl": 29, "sever": 29, "compon": 29, "datafram": 29, "bag": 29, "delai": 29, "decompos": 29, "fine": 29, "grain": 29, "higher": 29, "easier": 29, "visual": 29, "reason": 29, "newer": 29, "wherea": 29, "varieti": 29, "matur": 29, "influenc": 29, "some": 29, "util": 29, "continu": 29, "zappi": 29, "what": 29, "interven": 29, "wasn": 29, "concern": 29, "less": 29, "daunt": 29, "And": 29, "better": 29}, "objects": {"cubed": [[5, 0, 1, "", "Array"], [9, 0, 1, "", "Callback"], [10, 0, 1, "", "Spec"], [11, 0, 1, "", "TaskEndEvent"], [12, 2, 1, "", "apply_gufunc"], [13, 2, 1, "", "compute"], [14, 2, 1, "", "from_array"], [15, 2, 1, "", "from_zarr"], [16, 2, 1, "", "map_blocks"], [17, 2, 1, "", "measure_reserved_memory"], [23, 2, 1, "", "store"], [24, 2, 1, "", "to_zarr"], [25, 2, 1, "", "visualize"]], "cubed.Array": [[5, 1, 1, "", "__init__"], [6, 1, 1, "", "compute"], [7, 1, 1, "", "rechunk"], [8, 1, 1, "", "visualize"]], "cubed.Callback": [[9, 1, 1, "", "__init__"]], "cubed.Spec": [[10, 1, 1, "", "__init__"]], "cubed.TaskEndEvent": [[11, 1, 1, "", "__init__"]], "cubed.array_api": [[1, 2, 1, "", "arange"], [1, 2, 1, "", "asarray"], [1, 2, 1, "", "broadcast_to"], [1, 2, 1, "", "empty"], [1, 2, 1, "", "empty_like"], [1, 2, 1, "", "eye"], [1, 2, 1, "", "full"], [1, 2, 1, "", "full_like"], [1, 2, 1, "", "linspace"], [1, 2, 1, "", "ones"], [1, 2, 1, "", "ones_like"], [1, 2, 1, "", "zeros"], [1, 2, 1, "", "zeros_like"]], "cubed.random": [[18, 2, 1, "", "random"]], "cubed.runtime.executors.beam": [[19, 0, 1, "", "BeamDagExecutor"]], "cubed.runtime.executors.beam.BeamDagExecutor": [[19, 1, 1, "", "__init__"]], "cubed.runtime.executors.lithops": [[20, 0, 1, "", "LithopsDagExecutor"]], "cubed.runtime.executors.lithops.LithopsDagExecutor": [[20, 1, 1, "", "__init__"]], "cubed.runtime.executors.modal_async": [[21, 0, 1, "", "AsyncModalDagExecutor"]], "cubed.runtime.executors.modal_async.AsyncModalDagExecutor": [[21, 1, 1, "", "__init__"]], "cubed.runtime.executors.python": [[22, 0, 1, "", "PythonDagExecutor"]], "cubed.runtime.executors.python.PythonDagExecutor": [[22, 1, 1, "", "__init__"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"]}, "titleterms": {"api": [0, 1, 4], "refer": 0, "arrai": [0, 1, 4, 5, 6, 7, 8, 27], "io": 0, "chunk": 0, "specif": 0, "function": 0, "random": [0, 18], "number": 0, "gener": 0, "runtim": [0, 2, 4, 19, 20, 21, 22], "executor": [0, 19, 20, 21, 22], "python": [1, 22], "miss": 1, "from": 1, "cube": [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27], "differ": 1, "between": 1, "standard": 1, "comput": [2, 6, 13], "plan": 2, "memori": [2, 27], "execut": 2, "featur": 2, "contribut": 3, "develop": [3, 27], "design": 4, "storag": 4, "primit": 4, "oper": [4, 28], "core": 4, "rechunk": [7, 28], "visual": [8, 25], "callback": 9, "spec": 10, "taskendev": 11, "apply_gufunc": 12, "from_arrai": 14, "from_zarr": 15, "map_block": [16, 28], "measure_reserved_memori": 17, "beam": 19, "beamdagexecutor": 19, "lithop": 20, "lithopsdagexecutor": 20, "modal_async": 21, "asyncmodaldagexecutor": 21, "pythondagexecutor": 22, "store": 23, "to_zarr": 24, "get": 26, "start": 26, "instal": 26, "conda": 26, "pip": 26, "exampl": 26, "bound": 27, "serverless": 27, "distribut": 27, "n": 27, "dimension": 27, "process": 27, "motiv": 27, "document": 27, "For": 27, "user": 27, "depend": 28, "tree": 28, "elemwis": 28, "map_direct": 28, "blockwis": 28, "reduct": 28, "arg_reduct": 28, "relat": 29, "project": 29, "dask": 29, "previou": 29, "work": 29}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"API Reference": [[0, "api-reference"]], "Array": [[0, "array"]], "IO": [[0, "io"]], "Chunk-specific functions": [[0, "chunk-specific-functions"]], "Random number generation": [[0, "random-number-generation"]], "Runtime": [[0, "runtime"], [4, "runtime"]], "Executors": [[0, "executors"]], "Python Array API": [[1, "python-array-api"]], "Missing from Cubed": [[1, "missing-from-cubed"]], "Differences between Cubed and the standard": [[1, "differences-between-cubed-and-the-standard"]], "Computation": [[2, "computation"]], "Plan": [[2, "plan"]], "Memory": [[2, "memory"]], "Execution": [[2, "execution"]], "Runtime Features": [[2, "runtime-features"]], "Contributing": [[3, "contributing"]], "Development": [[3, "development"]], "Design": [[4, "design"]], "Storage": [[4, "storage"]], "Primitive operations": [[4, "primitive-operations"]], "Core operations": [[4, "core-operations"]], "Array API": [[4, "array-api"]], "cubed.Array": [[5, "cubed-array"]], "cubed.Array.compute": [[6, "cubed-array-compute"]], "cubed.Array.rechunk": [[7, "cubed-array-rechunk"]], "cubed.Array.visualize": [[8, "cubed-array-visualize"]], "cubed.Callback": [[9, "cubed-callback"]], "cubed.Spec": [[10, "cubed-spec"]], "cubed.TaskEndEvent": [[11, "cubed-taskendevent"]], "cubed.apply_gufunc": [[12, "cubed-apply-gufunc"]], "cubed.compute": [[13, "cubed-compute"]], "cubed.from_array": [[14, "cubed-from-array"]], "cubed.from_zarr": [[15, "cubed-from-zarr"]], "cubed.map_blocks": [[16, "cubed-map-blocks"]], "cubed.measure_reserved_memory": [[17, "cubed-measure-reserved-memory"]], "cubed.random.random": [[18, "cubed-random-random"]], "cubed.runtime.executors.beam.BeamDagExecutor": [[19, "cubed-runtime-executors-beam-beamdagexecutor"]], "cubed.runtime.executors.lithops.LithopsDagExecutor": [[20, "cubed-runtime-executors-lithops-lithopsdagexecutor"]], "cubed.runtime.executors.modal_async.AsyncModalDagExecutor": [[21, "cubed-runtime-executors-modal-async-asyncmodaldagexecutor"]], "cubed.runtime.executors.python.PythonDagExecutor": [[22, "cubed-runtime-executors-python-pythondagexecutor"]], "cubed.store": [[23, "cubed-store"]], "cubed.to_zarr": [[24, "cubed-to-zarr"]], "cubed.visualize": [[25, "cubed-visualize"]], "Getting Started": [[26, "getting-started"]], "Installation": [[26, "installation"]], "Conda": [[26, "conda"]], "Pip": [[26, "pip"]], "Example": [[26, "example"]], "Cubed": [[27, "cubed"]], "Bounded-memory serverless distributed N-dimensional array processing": [[27, "bounded-memory-serverless-distributed-n-dimensional-array-processing"]], "Motivation": [[27, "motivation"]], "Documentation": [[27, "documentation"]], "For users": [[27, null]], "For developers": [[27, null]], "Operations": [[28, "operations"]], "Dependency Tree": [[28, "dependency-tree"]], "elemwise": [[28, "elemwise"]], "map_blocks": [[28, "map-blocks"]], "map_direct": [[28, "map-direct"]], "blockwise": [[28, "blockwise"]], "rechunk": [[28, "rechunk"]], "reduction and arg_reduction": [[28, "reduction-and-arg-reduction"]], "Related Projects": [[29, "related-projects"]], "Dask": [[29, "dask"]], "Previous work": [[29, "previous-work"]]}, "indexentries": {"arange() (in module cubed.array_api)": [[1, "cubed.array_api.arange"]], "asarray() (in module cubed.array_api)": [[1, "cubed.array_api.asarray"]], "broadcast_to() (in module cubed.array_api)": [[1, "cubed.array_api.broadcast_to"]], "empty() (in module cubed.array_api)": [[1, "cubed.array_api.empty"]], "empty_like() (in module cubed.array_api)": [[1, "cubed.array_api.empty_like"]], "eye() (in module cubed.array_api)": [[1, "cubed.array_api.eye"]], "full() (in module cubed.array_api)": [[1, "cubed.array_api.full"]], "full_like() (in module cubed.array_api)": [[1, "cubed.array_api.full_like"]], "linspace() (in module cubed.array_api)": [[1, "cubed.array_api.linspace"]], "ones() (in module cubed.array_api)": [[1, "cubed.array_api.ones"]], "ones_like() (in module cubed.array_api)": [[1, "cubed.array_api.ones_like"]], "zeros() (in module cubed.array_api)": [[1, "cubed.array_api.zeros"]], "zeros_like() (in module cubed.array_api)": [[1, "cubed.array_api.zeros_like"]], "array (class in cubed)": [[5, "cubed.Array"]], "__init__() (cubed.array method)": [[5, "cubed.Array.__init__"]], "compute() (cubed.array method)": [[6, "cubed.Array.compute"]], "rechunk() (cubed.array method)": [[7, "cubed.Array.rechunk"]], "visualize() (cubed.array method)": [[8, "cubed.Array.visualize"]], "callback (class in cubed)": [[9, "cubed.Callback"]], "__init__() (cubed.callback method)": [[9, "cubed.Callback.__init__"]], "spec (class in cubed)": [[10, "cubed.Spec"]], "__init__() (cubed.spec method)": [[10, "cubed.Spec.__init__"]], "taskendevent (class in cubed)": [[11, "cubed.TaskEndEvent"]], "__init__() (cubed.taskendevent method)": [[11, "cubed.TaskEndEvent.__init__"]], "apply_gufunc() (in module cubed)": [[12, "cubed.apply_gufunc"]], "compute() (in module cubed)": [[13, "cubed.compute"]], "from_array() (in module cubed)": [[14, "cubed.from_array"]], "from_zarr() (in module cubed)": [[15, "cubed.from_zarr"]], "map_blocks() (in module cubed)": [[16, "cubed.map_blocks"]], "measure_reserved_memory() (in module cubed)": [[17, "cubed.measure_reserved_memory"]], "random() (in module cubed.random)": [[18, "cubed.random.random"]], "beamdagexecutor (class in cubed.runtime.executors.beam)": [[19, "cubed.runtime.executors.beam.BeamDagExecutor"]], "__init__() (cubed.runtime.executors.beam.beamdagexecutor method)": [[19, "cubed.runtime.executors.beam.BeamDagExecutor.__init__"]], "lithopsdagexecutor (class in cubed.runtime.executors.lithops)": [[20, "cubed.runtime.executors.lithops.LithopsDagExecutor"]], "__init__() (cubed.runtime.executors.lithops.lithopsdagexecutor method)": [[20, "cubed.runtime.executors.lithops.LithopsDagExecutor.__init__"]], "asyncmodaldagexecutor (class in cubed.runtime.executors.modal_async)": [[21, "cubed.runtime.executors.modal_async.AsyncModalDagExecutor"]], "__init__() (cubed.runtime.executors.modal_async.asyncmodaldagexecutor method)": [[21, "cubed.runtime.executors.modal_async.AsyncModalDagExecutor.__init__"]], "pythondagexecutor (class in cubed.runtime.executors.python)": [[22, "cubed.runtime.executors.python.PythonDagExecutor"]], "__init__() (cubed.runtime.executors.python.pythondagexecutor method)": [[22, "cubed.runtime.executors.python.PythonDagExecutor.__init__"]], "store() (in module cubed)": [[23, "cubed.store"]], "to_zarr() (in module cubed)": [[24, "cubed.to_zarr"]], "visualize() (in module cubed)": [[25, "cubed.visualize"]]}})