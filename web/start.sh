#!/bin/bash

echo "🚀 Starting FedLearnDR Application..."
echo ""
echo "Starting Backend API on port 3000..."
cd server && npm start &
BACKEND_PID=$!

echo "Waiting for backend to start..."
sleep 3

echo ""
echo "Starting Frontend on port 5173..."
cd ..
npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================="
echo "✅ Application is starting!"
echo "========================================="
echo ""
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait
