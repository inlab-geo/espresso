.. raw:: html

    <div class="api-module">

Utility Functions
-----------------

.. raw:: html

    <hr>

.. automodule:: {{ fullname }}

.. {% block classes %}
.. {% if classes %}
.. .. rubric:: Classes

.. .. autosummary::
..   :toctree: ./
.. {% for item in classes %}
..   {{ fullname }}.{{ item }}
.. {% endfor %}
.. {% endif %}
.. {% endblock %}


{% block functions %}
{% if functions %}
.. rubric:: Functions

{% for item in functions %}
.. automethod:: {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}
{% endblock %}


.. {% block exceptions %}
.. {% if exceptions %}
.. .. rubric:: Exceptions

.. .. autosummary::
..   :toctree: ./
.. {% for item in exceptions %}
..   {{ fullname }}.{{ item }}
.. {% endfor %}
.. {% endif %}
.. {% endblock %}

.. raw:: html

    </div>
