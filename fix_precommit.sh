#!/bin/bash

# Script para ejecutar pre-commit correctamente

echo "🔧 Ejecutando pre-commit fixes..."
echo ""

cd /Users/joaquintschopp/proyectos_maestria/VC-ARN

# Ejecutar pre-commit
pre-commit run --all-files

# Verificar resultado
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pre-commit pasó exitosamente!"
    echo ""
    echo "Ahora puedes hacer commit:"
    echo "  cd TP-FINAL/src"
    echo "  git add ."
    echo "  git commit -m 'Proyecto CNN - pre-commit fixes'"
else
    echo ""
    echo "⚠️  Pre-commit encontró problemas"
    echo "Ejecutando manualmente para resolver..."
    echo ""
    
    cd TP-FINAL/src
    
    echo "1️⃣  Ejecutando isort..."
    python3 -m isort .
    
    echo "2️⃣  Ejecutando black..."
    python3 -m black .
    
    echo "3️⃣  Ejecutando ruff..."
    python3 -m ruff check .
    
    echo ""
    echo "✅ Herramientas ejecutadas manualmente"
    echo "Intenta nuevamente con: pre-commit run --all-files"
fi
