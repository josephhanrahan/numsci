python_sources = [
    '__init__.py',
]

py = import('python').find_installation()

py.install_sources(
    python_sources,
    pure: false,
    subdir: 'numsci',
)

subdir('optimize')