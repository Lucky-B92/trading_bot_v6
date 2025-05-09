// Configuração global do Chart.js
Chart.defaults.color = '#ECEFF1';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';

document.addEventListener('DOMContentLoaded', function() {
    // Carrega dados iniciais
    loadEquityChart();
    loadClosedTradesChart();
    loadOpenTradesChart();
    loadLogsTable(); // Nova função
    setupAutoRefresh();
});

// 1. Gráfico de Evolução da Banca
// Referência global para o gráfico
let equityChart = null;

// Função para carregar ou atualizar o gráfico
function loadEquityChart() {
    fetch('/get_equity_history')
        .then(handleResponse)
        .then(data => {
            const ctx = document.getElementById('equity-chart');
            if (!ctx) return;

            const labels = data.map(item => {
                const date = new Date(item.date);
                const day = String(date.getDate()).padStart(2, '0');
                const month = String(date.getMonth() + 1).padStart(2, '0');
                return `${day}/${month}`;
            });            
            const equityData = data.map(item => item.equity);

            if (equityChart) {
                // Atualizar dados do gráfico existente
                equityChart.data.labels = labels;
                equityChart.data.datasets[0].data = equityData;
                equityChart.update();
            } else {
                // Criar um novo gráfico
                equityChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Banca (USDT)',
                            data: equityData,
                            borderColor: '#03DAC6',
                            backgroundColor: 'rgba(3, 218, 198, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: getChartOptions('Valor da Banca ao Longo do Tempo')
                });
            }
        })
        .catch(err => {
            console.error('Erro ao atualizar gráfico de equity:', err);
        });
}


// 2. Gráfico de Trades Fechados
function loadClosedTradesChart() {
    fetch('/get_trades?status=closed')
        .then(handleResponse)
        .then(data => {
            const ctx = document.getElementById('closed-trades-chart');
            if (!ctx) return;

            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.map((_, i) => `Trade ${i+1}`),
                    datasets: [{
                        label: 'Resultado (USDT)',
                        data: data.map(trade => trade.pnl),
                        backgroundColor: data.map(trade => 
                            trade.pnl >= 0 ? 'rgba(3, 218, 198, 0.7)' : 'rgba(207, 102, 121, 0.7)'
                        ),
                        borderColor: data.map(trade => 
                            trade.pnl >= 0 ? '#03DAC6' : '#CF6679'
                        ),
                        borderWidth: 1
                    }]
                },
                options: getChartOptions('Performance por Trade', false)
            });
        })
        .catch(handleError('closed-trades-chart'));
}

// 3. Gráfico de Trades Abertos (Novo)
function loadOpenTradesChart() {
    fetch('/get_trades?status=open')
        .then(handleResponse)
        .then(data => {
            const ctx = document.getElementById('open-trades-chart');
            if (!ctx || data.length === 0) {
                ctx.closest('.card').style.display = 'none';
                return;
            }

            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.map(trade => trade.symbol),
                    datasets: [{
                        data: data.map(trade => trade.amount),
                        backgroundColor: [
                            '#03DAC6', '#BB86FC', '#CF6679', 
                            '#018786', '#3700B3', '#6200EE'
                        ]
                    }]
                },
                options: getChartOptions('Trades Abertos por Ativo', false)
            });
        })
        .catch(handleError('open-trades-chart'));
}

// Funções auxiliares
function getChartOptions(title, showLegend = true) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: showLegend,
                position: 'top',
                labels: {
                    font: {
                        size: 12
                    }
                }
            },
            title: {
                display: !!title,
                text: title,
                color: '#ECEFF1',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            x: {
                grid: {
                    display: false
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)'
                }
            }
        }
    };
}

function handleResponse(response) {
    if (!response.ok) throw new Error('Erro na rede');
    return response.json();
}

function handleError(chartId) {
    return error => {
        console.error(`Erro no gráfico ${chartId}:`, error);
        const element = document.getElementById(chartId);
        if (element) {
            element.innerHTML = `<div class="chart-error">Erro ao carregar dados</div>`;
        }
    };
}

// Nova função para carregar a tabela de logs
function loadLogsTable() {
    fetch('/get_system_logs')
        .then(response => {
            if (!response.ok) throw new Error('Erro ao carregar logs');
            return response.json();
        })
        .then(data => {
            const container = document.getElementById('logs-container');
            if (!container) return;

            container.innerHTML = data.length > 0 
                ? generateLogsTableHTML(data) 
                : '<div class="no-logs">Nenhum log disponível</div>';
        })
        .catch(error => {
            console.error('Erro na tabela de logs:', error);
            const container = document.getElementById('logs-container');
            if (container) {
                container.innerHTML = '<div class="logs-error">Erro ao carregar logs</div>';
            }
        });
}

// Nova função auxiliar para gerar HTML da tabela
function generateLogsTableHTML(logs) {
    return `
        <table class="logs-table">
            <thead>
                <tr>
                    <th>Data/Hora</th>
                    <th>Tipo</th>
                    <th>Mensagem</th>
                </tr>
            </thead>
            <tbody>
                ${logs.map(log => `
                    <tr class="log-row log-${log.type.toLowerCase()}">
                        <td>${new Date(log.timestamp).toLocaleString()}</td>
                        <td><span class="log-badge">${log.type}</span></td>
                        <td>${log.message}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function setupAutoRefresh() {
    // Atualiza a cada 30 segundos
    setInterval(() => {
        if (document.hidden) return;
        loadEquityChart();
        loadClosedTradesChart();
        loadOpenTradesChart();
        loadLogsTable(); // Nova chamada
    }, 30000);
}