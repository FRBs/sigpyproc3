{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: {{ fullname }}

{# ---- Classes Summary ---- #}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{%   for item in classes %}
   {{ item }}
{%   endfor %}
{% endif %}

{# ---- Functions Summary ---- #}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{%   for item in functions %}
   {{ item }}
{%   endfor %}
{% endif %}

{# ---- Module Attributes Summary ---- #}
{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
{%   for item in attributes %}
   {{ item }}
{%   endfor %}
{% endif %}


{# ---- Module Attributes Details ---- #}
{% if attributes %}
{%   for item in attributes %}
.. autodata:: {{ item }}
   :no-value:

{%   endfor %}
{% endif %}


{# ---- Class Details ---- #}
{% if classes %}
{%   for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :show-inheritance:

{%   endfor %}
{% endif %}

{# ---- Function Details ---- #}
{% if functions %}
{%   for item in functions %}
.. autofunction:: {{ item }}

{%   endfor %}
{% endif %}