cc = meson.get_compiler('c')
m_dep = cc.find_library('m')

py = import('python').find_installation()

shared_library(
  'minpack', 
  ['__minpack.c', 'residual.c'],
  install : true,
  c_args : ['-O3'],
  dependencies : [m_dep]
)