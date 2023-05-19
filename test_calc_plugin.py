from medex.services.importer import AdditionPlugin

data = {
    "key1": 20,
    "key2": 40
}


plugin = AdditionPlugin()
result = plugin.calculate(data)

print(result)
