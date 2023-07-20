.. raw:: html

    <div class="api-module">

    <hr>

.. automodule:: {{ fullname }}

{% block classes %}
{% if classes %}
.. rubric:: Classes
{% for item in classes %}
.. autoclass:: {{ fullname }}.{{ item }}
    :members:
    :inherited-members:
    :show-inheritance:
    :undoc-members:
    :exclude-members: __init__, __weakref__
    :special-members: __eq__, __ne__, __lt__, __le__, __gt__, __ge__, __hash__
{% endfor %}
{% endif %}
{% endblock %}


{% block functions %}
{% if functions %}
.. rubric:: Functions

{% for item in functions %}
.. automethod:: {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}
{% endblock %}


{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions
{% for item in exceptions %}
.. autoexception:: {{ fullname }}.{{ item }}
{% endfor %}
{% endif %}
{% endblock %}

.. raw:: html

    </div>
