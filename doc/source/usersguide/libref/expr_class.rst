.. currentmodule:: tables

General purpose expression evaluator class
==========================================

The Expr class
--------------
.. autoclass:: Expr

..  These are defined in the class docstring.
    Expr instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: Expr.append_mode
    .. autoattribute:: Expr.maindim
    .. autoattribute:: Expr.names
    .. autoattribute:: Expr.out
    .. autoattribute:: Expr.o_start
    .. autoattribute:: Expr.o_stop
    .. autoattribute:: Expr.o_step
    .. autoattribute:: Expr.shape
    .. autoattribute:: Expr.values


Expr methods
~~~~~~~~~~~~
.. automethod:: Expr.eval

.. automethod:: Expr.set_inputs_range

.. automethod:: Expr.set_output

.. automethod:: Expr.set_output_range


Expr special methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: Expr.__iter__
