const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');

// Middleware to parse JSON and URL-encoded data
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

let queries = []; // Array to store the user's queries

// Endpoint to handle the user's query submission
app.post('/submit-query', (req, res) => {
    const query = req.body.query;
    if (query) {
        queries.push(query); // Store the query in history
    }
    res.json({ success: true, queries }); // Send back the updated queries list
});

// Serve the index.html file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

// Start the server on port 3003
const port = 3004;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
