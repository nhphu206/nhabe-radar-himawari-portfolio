# Licensing and third-party software

## Project license

Copyright (C) 2026 Nguyen Hoai Phu.

The original project source code, scripts, tests, configuration, and accompanying
project-authored documentation are licensed under the GNU General Public License,
version 3 or (at your option) any later version (**GPL-3.0-or-later**).
See [LICENSE](LICENSE) for the full terms. This software is provided without warranty.

This grant does not cover raw radar observations, Himawari observations, generated
research datasets, personal reports, or third-party materials. No permission to
redistribute those materials is granted by this repository's license. Small research
validation snapshots in `docs/validation/` describe results; the source-code license
does not grant rights to the underlying observation dataset.

## Direct dependencies

Dependencies retain their own copyright notices and license terms. This table is a
summary of the direct dependencies, not a complete license inventory of every component
installed in a container. Bundled and transitive components can have additional terms.

| Dependency | Pinned version | Upstream license reference |
| --- | --- | --- |
| Py-ART | 2.2.5 | [BSD-3-Clause](https://github.com/ARM-DOE/pyart/blob/main/LICENSE.txt) |
| SatPy | 0.60.0 | [GPL-3.0-or-later](https://github.com/pytroll/satpy/blob/v0.60.0/LICENSE.txt) |
| Pyresample | 1.35.0 | [LGPLv3 license text](https://github.com/pytroll/pyresample/blob/v1.35.0/LICENSE.txt) |
| NumPy | 2.5.2 | [BSD and bundled third-party notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| PyYAML | 6.0.3 | [MIT](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| aiohttp | 3.14.3 | [Apache-2.0 and bundled notices](https://github.com/aio-libs/aiohttp/blob/master/LICENSE.txt) |
| PyTorch | 2.14.1 | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| FastAPI | 0.162.0 | [MIT](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| Uvicorn | 0.54.0 | [BSD-3-Clause](https://github.com/Kludex/uvicorn/blob/master/LICENSE.md) |
| HTTPX2 | 2.13.0 | [BSD-3-Clause](https://github.com/pydantic/httpx/blob/main/LICENSE.md) |

The HSD timing reader imports SatPy's header dtype definitions rather than including
a copied SatPy reader implementation. It uses private API names, so compatibility is
version-dependent. Radar decoding/gridding and satellite calibration/resampling are
performed by the upstream libraries; this project does not claim authorship of them.

## Container distribution

Docker builds install third-party packages and copy the installed environment, including
package license files, into the runtime. The project LICENSE and this notice are also
included. A successful build is not proof of complete redistribution compliance.

Before distributing a built image, review all bundled Python and operating-system
components, retain required notices, and satisfy any applicable corresponding-source
requirements. Provide the exact project source and build instructions for the released
version, and a compliant source-distribution mechanism for components that require it;
an upstream homepage link alone is not assumed sufficient. No container registry
publication or complete image license/source audit has been performed for this project.

This file is an engineering record, not a legal opinion. The full license terms govern.
