import pathlib


class paths:
    module_dir = pathlib.Path(__file__).resolve().parent
    top_level = module_dir.parent.parent
    frontend = top_level / 'src' / 'frontend'
    ui = frontend / 'build'
    logdir = top_level / 'logs'
    data = top_level / 'data'
    analyzed = data / 'analyzed'
    gruntwork = data / 'gruntwork'

    figures = top_level / 'reports' / 'figures'

    cache = top_level / 'cache'
    models = top_level.parent / 'models'
    scripts = top_level / 'scripts'

    old_code = frontend / 'src' / 'old_versions'
