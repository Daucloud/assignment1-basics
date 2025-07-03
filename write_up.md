# Unicode 1
1. `'\x00'`
2. `print(chr(0))` outputs nothing but an empty line.
3. It simply disappears when printed but keeps the original '\x00' code if not.

# Unicode 2
1. UTF-8 will encode the string in the most shortest length, which may help save the cost.
2. One byte doesn't correponds to one Unicode character necessarily. Any unicode character encoded by multiple bytes in UTF-8 is a counter-example, such as the Chinese character '坤'.
3. `b'\xe4\xbd'`
