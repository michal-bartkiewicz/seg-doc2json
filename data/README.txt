Private local input layout (these files are git-ignored):

  data/documents/<document-type>/*.{pdf,txt}  documents to process
  data/documents/inbox/*.{pdf,txt}            files classified locally with regex
  data/samples/<document-type>/*.{pdf,txt}    labeled examples for Gemini
  data/expected.json                          per-type schema examples and 3 labeled samples per type

See the top-level README.md for the expected.json format and run command.
