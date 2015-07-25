import pypandoc

#converts markdown to reStructured
z = pypandoc.convert('README.rst.md','rst',format='markdown')

#writes converted file
with open('README.rst','w') as outfile:
        outfile.write(z)
