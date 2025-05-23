document.addEventListener('DOMContentLoaded', function() {
    // Controle do bot
    const toggleBotBtn = document.getElementById('toggle-bot');
    const botStatusIndicator = document.querySelector('.status-indicator');
    const botStatusText = document.querySelector('.status-text');
    
    toggleBotBtn.addEventListener('click', function() {
        const isActive = botStatusIndicator.classList.contains('active');
        
        fetch('/toggle_bot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ active: !isActive })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (!isActive) {
                    botStatusIndicator.classList.remove('inactive');
                    botStatusIndicator.classList.add('active');
                    botStatusText.textContent = 'Ativo';
                    toggleBotBtn.textContent = 'Parar Bot';
                } else {
                    botStatusIndicator.classList.remove('active');
                    botStatusIndicator.classList.add('inactive');
                    botStatusText.textContent = 'Inativo';
                    toggleBotBtn.textContent = 'Iniciar Bot';
                }
            }
        });
    });
    
    // Atualizar status periodicamente
    function updateStatus() {
        fetch('/get_status')
            .then(response => response.json())
            .then(data => {
                // Status do robô
                document.getElementById('bot-active-status').textContent = 
                    data.status.active ? 'Ativo' : 'Inativo';
                
                // Último ciclo
                document.getElementById('last-cycle').textContent = 
                    data.last_cycle || 'N/A';  // Agora é um campo independente
    
                // Operações ativas
                document.getElementById('active-operations').textContent = 
                    data.status.current_operations;
                
                // Período Bloqueado
                const blockedElement = document.getElementById('blocked-time');
                if (data.blocked) {
                    blockedElement.textContent = 'Sim (20h-2h)';
                    blockedElement.classList.add('warning');
                } else {
                    blockedElement.textContent = 'Não';
                    blockedElement.classList.remove('warning');
                }
            })
            .catch(err => {
                console.error('Erro ao atualizar status:', err);
            });
    }
    
    // Função para atualizar a tabela de operações abertas
    function updateOpenTradesTable() {
    fetch('/get_trades?status=open')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector('.open-trades-table tbody');
            tableBody.innerHTML = '';

            if (data.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="11">Nenhuma operação aberta</td></tr>';
                return;
            }

            data.forEach(trade => {
                const profitClass = trade.profit_pct > 0 ? 'positive' : 'negative';
                const row = `
                    <tr>
                        <td>${trade.symbol}</td>
                        <td>${trade.side}</td>
                        <td>${trade.open_price?.toFixed(4) || ''}</td>
                        <td>${trade.current_price?.toFixed(4) || ''}</td>
                        <td class="${profitClass}">${trade.profit_pct?.toFixed(2)}%</td>
                        <td class="${profitClass}">${trade.pnl?.toFixed(2)}</td>
                        <td>${trade.stop_loss?.toFixed(2) || ''}</td>
                        <td>${trade.potential_profit?.toFixed(2) || ''}</td>
                        <td>${trade.potential_loss?.toFixed(2) || ''}</td>
                        <td>${trade.expected_duration || ''}</td>
                        <td>
                            <button onclick="closeTrade('${trade.symbol}')" class="btn btn-danger btn-sm">Vender</button>
                        </td>
                    </tr>
                `;
                tableBody.insertAdjacentHTML('beforeend', row);
            });
        })
        .catch(err => console.error('Erro ao atualizar operações abertas:', err));
}

    

    // Função para encerrar uma operação
    function closeTrade(tradeId) {
        fetch('/close_trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ id: tradeId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`Trade ${tradeId} encerrado com sucesso.`);
                updateOpenTradesTable();
            } else {
                console.error('Erro ao encerrar trade:', data.error);
            }
        });
    }

    // Torna a função acessível globalmente
    window.closeTrade = closeTrade;


    // Intervalos de atualização
    setInterval(updateStatus, 5000);
    updateStatus();

    setInterval(updateOpenTradesTable, 5000);
    updateOpenTradesTable();
    
    // Atualizar insights
    document.getElementById('refresh-insights').addEventListener('click', function () {
        const insightsContent = document.getElementById('insights-content');
        insightsContent.innerHTML = "Gerando insights com IA...";

        fetch('/generate_insights')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Após gerar, faz nova chamada para buscar os dados
                    fetch('/get_insights')
                        .then(response => response.json())
                        .then(data => {
                            if (Array.isArray(data) && data.length > 0) {
                                let insightsHtml = '';
                                data.forEach(insight => {
                                    insightsHtml += `<p>${insight.content}</p>`;
                                });
                                insightsContent.innerHTML = insightsHtml;
                            } else {
                                insightsContent.innerHTML = "Nenhum insight disponível.";
                            }
                        })
                        .catch(err => {
                            console.error('Erro ao buscar insights:', err);
                            insightsContent.innerHTML = "Erro ao carregar insights.";
                        });
                } else {
                    insightsContent.innerHTML = "Erro ao gerar insights: " + data.error;
                }
            })
            .catch(err => {
                console.error('Erro ao gerar insights:', err);
                insightsContent.innerHTML = "Erro ao gerar insights.";
            });
    });

    

    // Atualizar configurações de timeframes e indicadores
    document.addEventListener('DOMContentLoaded', function() {
        const settingsForm = document.getElementById('timeframe-settings-form');

        if (settingsForm) {
            settingsForm.addEventListener('submit', function(e) {
                e.preventDefault();

                const primaryTimeframe = document.getElementById('primary-timeframe').value;
                const confirmationTimeframe = document.getElementById('confirmation-timeframe').value;
                const emaShort = parseInt(document.getElementById('ema-short').value);
                const rsiPeriod = parseInt(document.getElementById('rsi-period').value);
                const macdFast = parseInt(document.getElementById('macd-fast').value);
                const macdSlow = parseInt(document.getElementById('macd-slow').value);
                const macdSignal = parseInt(document.getElementById('macd-signal').value);

                const settings = {
                    primary: primaryTimeframe,
                    confirmation: confirmationTimeframe,
                    ema_short_period: emaShort,
                    rsi_period: rsiPeriod,
                    macd_fast: macdFast,
                    macd_slow: macdSlow,
                    macd_signal: macdSignal
                };

                fetch('/update_settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ type: 'model', settings })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showToast('Configurações atualizadas com sucesso!', 'success');
                    } else {
                        showToast(`Erro: ${data.error}`, 'danger');
                    }
                })
                .catch(err => {
                    console.error('Erro ao atualizar configurações:', err);
                });
            });
        }
    });

    // Carregar configurações ao abrir a página de configurações
    document.addEventListener('DOMContentLoaded', function() {
        fetch('/get_settings')
            .then(response => response.json())
            .then(data => {
                if (data.timeframes) {
                    document.getElementById('primary-timeframe').value = data.timeframes.primary;
                    document.getElementById('confirmation-timeframe').value = data.timeframes.confirmation;
                }

                if (data.model) {
                    document.getElementById('ema-short').value = data.model.ema_short;
                    document.getElementById('rsi-period').value = data.model.rsi;
                    document.getElementById('macd-fast').value = data.model.macd_fast;
                    document.getElementById('macd-slow').value = data.model.macd_slow;
                    document.getElementById('macd-signal').value = data.model.macd_signal;
                }
            })
            .catch(err => {
                console.error('Erro ao carregar configurações:', err);
            });
    });


});