/**
 * Reusable data table with optional empty state.
 * @param {string[]} columns  - Array of column header strings
 * @param {Array}    rows     - Array of row arrays (each row = array of cell content)
 * @param {string}   [emptyMessage]
 */
export default function DataTable({ columns, rows, emptyMessage = 'No data available.' }) {
  return (
    <div className="table-wrapper">
      <table aria-label="Data table">
        <thead>
          <tr>
            {columns.map((col, i) => (
              <th key={i} scope="col">{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                {emptyMessage}
              </td>
            </tr>
          ) : (
            rows.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci}>{cell}</td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
