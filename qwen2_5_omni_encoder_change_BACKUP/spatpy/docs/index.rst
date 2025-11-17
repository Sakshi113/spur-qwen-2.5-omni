spatpy
=======================================================

.. centered:: |release|

``spatpy`` is a collection of utilities for working with spatial audio, including but not limited to:

- 3D geometry: :obj:`spatpy.geometry`, :obj:`spatpy.placement`.
- Ambisonic channel formats: :obj:`spatpy.ambisonics`, :obj:`spatpy.wxy`, and :obj:`spatpy.beehive`
- Frequency response measurement, modelling and simulation: :obj:`spatpy.device_model`, :obj:`spatpy.room_model`
- Automatic EQ matching: :obj:`spatpy.eq`
- Codecs and binauralisation: :obj:`spatpy.lc3`, :obj:`spatpy.binaural`


Install for development
-----------------------

.. code:: bash

    pip3 install -e git+ssh://git@gitlab-sfo.dolby.net/capture/spatpy.git@main#egg=spatpy --src .


Install from devpi
-----------------------
.. code:: bash

    pip3 install spatpy --extra-index https://devpi.dolby.net/capture/main

.. seealso::
    The `README for this project on gitlab <https://gitlab-sfo.dolby.net/capture/spatpy>`_.

.. toctree::
   :maxdepth: 3
   
   Console Scripts <cmdline.rst>
   API Reference <_apidoc/spatpy.rst>