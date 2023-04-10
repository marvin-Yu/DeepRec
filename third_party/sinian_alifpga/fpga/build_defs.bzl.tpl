# Macros for building FPGA code.
def sinian_alifpga_is_configured():
    return %{sinian_alifpga_is_configured}%

def if_sinian_alifpga_is_configured(x):
    if sinian_alifpga_is_configured():
      return x
    return []

def sinian_alifpga_copts():
    if sinian_alifpga_is_configured():
        return ['-DSINIAN_ALIFPGA=1', '-L/usr/local/lib/', '-L/usr/lib/', '-lAliDLA']
    else:
        return []
