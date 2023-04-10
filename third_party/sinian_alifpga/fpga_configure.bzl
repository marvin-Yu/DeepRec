# -*- Python -*-

def _enable_fpga(repository_ctx):
  if "TF_NEED_FPGA" in repository_ctx.os.environ:
    enable_fpga = repository_ctx.os.environ["TF_NEED_FPGA"].strip()
    return enable_fpga == "1"
  return False

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/sinian_alifpga/%s.tpl" % tpl),
      substitutions)

def _fpga_autoconf_impl(repository_ctx):
    _tpl(repository_ctx, "fpga:build_defs.bzl", {
        "%{sinian_alifpga_is_configured}%" : "True" if _enable_fpga(repository_ctx) else "False"
    })
    _tpl(repository_ctx, "fpga:BUILD", {})

sinian_alifpga_configure = repository_rule(
    implementation = _fpga_autoconf_impl,
    local = True,
)
