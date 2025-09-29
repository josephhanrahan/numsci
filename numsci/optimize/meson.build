cc = meson.get_compiler('c')
m_dep = cc.find_library('m')

py = import('python').find_installation()

shared_library(
  'minpack', 
  ['__minpack.c', 'residual.c'],
  install : true,
  install_dir : py.get_install_dir() / 'numsci/optimize',
  c_args : ['-O3'],
  dependencies : [m_dep]
)

python_sources = [
  '__init__.py',
  'minpackcore.py'
]

py.install_sources(
  python_sources,
  subdir: 'numsci/optimize'
)