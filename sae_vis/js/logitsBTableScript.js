// ! Using `DATA.logitsBTableData` to fill in Model B pos/neg logit tables
function setupLogitTablesB(logitsTableId, tablesData, tableMetaData) {
    // Fetch table data
    const tableData = tablesData[tableMetaData.dataKey];

    // Select the table container by its ID
    const tablesContainer = d3.select(`#${logitsTableId}`);

    // Select or create table container
    const tableId = `${tableMetaData.class}-${logitsTableId}`;
    const tableContainer = tablesContainer.select(`#${tableId}`);
    
    // If this table container doesn't exist, create it
    if (tableContainer.empty()) {
        const section = tablesContainer.append("div").attr("class", tableMetaData.class);
        section.append("h4").html(tableMetaData.title);
        section.append("table").attr("id", tableId).attr("class", "table-left");
    }

    // Get table, clear it, and add new data
    const table = tablesContainer.select(`#${tableId}`);
    table.selectAll('tr').remove();

    // Bind logits data to rows
    const rows = table.selectAll('tr')
        .data(tableData)
        .enter()
        .append('tr');
    
    // Append token cell
    rows.append('td')
        .attr('class', 'left-aligned')
        .append('code')
        .style('background-color', d => d.color)
        .text(d => d.symbol);
    
    // Append value cell
    rows.append('td')
        .attr('class', 'right-aligned')
        .text(d => d.value.toFixed(2));
}

// Define metadata for Model B tables
const logitTablesMetaDataB = [
    {title: "MODEL B NEGATIVE LOGITS", dataKey: "negLogits", class: "negative"},
    {title: "MODEL B POSITIVE LOGITS", dataKey: "posLogits", class: "positive"},
];

// Create Model B tables
Object.entries(DATA.logitsBTableData).forEach(([suffix, tablesData]) => {
    const logitsTableId = `logits-table-B-${suffix}`;  // Note the -B- in the ID
    logitTablesMetaDataB.forEach(tableMetaData => {
        setupLogitTablesB(logitsTableId, tablesData, tableMetaData);
    });
});