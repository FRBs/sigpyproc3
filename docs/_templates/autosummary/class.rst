{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

