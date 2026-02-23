{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
{% for item in attributes %}
.. autodata:: {{ item }}
   :no-value:

{% endfor %}
{% endif %}

{% if classes %}
{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :show-inheritance:

{% endfor %}
{% endif %}

{% if functions %}
{% for item in functions %}
.. autofunction:: {{ item }}

{% endfor %}
{% endif %}