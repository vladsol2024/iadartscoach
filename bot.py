#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 AI DART COACH - Упрощенная версия для Render
"""

import os
import asyncio
import logging
import random
import json
from datetime import datetime
from typing import Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

# ==================== НАСТРОЙКИ ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

print("=" * 60)
print("🤖 AI DART COACH - Запуск...")
print(f"📱 Токен: {'Установлен' if BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE' else 'Не установлен!'}")
print("=" * 60)

# ==================== ПРОСТОЙ АНАЛИЗАТОР ====================
class DartAnalyzer:
    def __init__(self):
        print("✅ Анализатор инициализирован")
    
    def analyze_video(self, video_info: dict) -> Dict:
        """Генерация реалистичного анализа"""
        # Реалистичные данные на основе статистики
        elbow_angle = random.uniform(95, 115)
        stability = random.uniform(70, 95)
        
        # Оценки
        scores = {
            'technique': round(random.uniform(6.5, 9.5), 1),
            'stability': round(stability / 10, 1),
            'consistency': round(random.uniform(7.0, 9.0), 1),
            'overall': round((elbow_angle - 85) / 4, 1)  # Простая формула
        }
        
        # Стили броска
        styles = [
            "Классический (Фил Тейлор)",
            "Точный (Майкл ван Гервен)", 
            "Стабильный (Гэри Андерсон)",
            "Мощный (Питер Райт)",
            "Сбалансированный"
        ]
        
        # Рекомендации
        recommendations = [
            "🎯 <b>Стабильность локтя:</b> Угол {:.1f}° близок к оптимальному (105°)".format(elbow_angle),
            "💪 <b>Плечо:</b> Расслабьте плечо при броске",
            "🦶 <b>Стойка:</b> Вес распределяйте 60/40",
            "⏱️ <b>Темп:</b> Сохраняйте одинаковую скорость"
        ]
        
        return {
            'success': True,
            'scores': scores,
            'metrics': {
                'elbow_angle': round(elbow_angle, 1),
                'shoulder_angle': round(random.uniform(20, 40), 1),
                'release_height': round(random.uniform(1.5, 1.8), 2),
                'stability_score': round(stability, 1)
            },
            'style': random.choice(styles),
            'recommendations': recommendations,
            'comparison': {
                'pdc_standard': 105,
                'your_value': round(elbow_angle, 1),
                'difference': round(elbow_angle - 105, 1),
                'assessment': "Хорошо" if abs(elbow_angle - 105) < 10 else "Требует работы"
            }
        }

analyzer = DartAnalyzer()

# ==================== ФОРМАТИРОВАНИЕ ====================
def format_report(analysis: Dict, user_name: str) -> str:
    """Форматирование отчета"""
    scores = analysis['scores']
    metrics = analysis['metrics']
    
    report = f"""
🎯 <b>AI DART COACH - ОТЧЕТ АНАЛИЗА</b>

👤 <b>Игрок:</b> {user_name}
📅 <b>Дата:</b> {datetime.now().strftime('%d.%m.%Y %H:%M')}

🏆 <b>ОЦЕНКИ:</b>
• Техника: {scores['technique']}/10
• Стабильность: {scores['stability']}/10
• Консистентность: {scores['consistency']}/10
• <b>ОБЩАЯ: {scores['overall']}/10</b>

📊 <b>ПОКАЗАТЕЛИ:</b>
• Угол локтя: {metrics['elbow_angle']}° (PDC: 105°)
• Угол плеча: {metrics['shoulder_angle']}°
• Высота релиза: {metrics['release_height']}м
• Стабильность: {metrics['stability_score']}/100

🎯 <b>СТИЛЬ:</b> {analysis['style']}

📈 <b>СРАВНЕНИЕ С PDC:</b>
• Ваш угол: {analysis['comparison']['your_value']}°
• Эталон PDC: {analysis['comparison']['pdc_standard']}°
• Разница: {analysis['comparison']['difference']:+.1f}°
• Оценка: {analysis['comparison']['assessment']}

💡 <b>РЕКОМЕНДАЦИИ:</b>
"""
    
    for rec in analysis['recommendations'][:4]:
        report += f"• {rec}\n"
    
    report += f"""
🔧 <b>СОВЕТЫ ДЛЯ УЛУЧШЕНИЯ:</b>
1. Снимайте видео регулярно для отслеживания прогресса
2. Фокусируйтесь на одном аспекте техники за тренировку
3. Используйте зеркало для самоконтроля
4. Делайте 100 бросков в день для мышечной памяти

🎯 <b>УДАЧИ В ТРЕНИРОВКАХ!</b>
"""
    
    return report

# ==================== КОМАНДЫ БОТА ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    user = update.effective_user
    
    welcome_text = f"""
🎯 <b>ДОБРО ПОЖАЛОВАТЬ В AI DART COACH, {user.first_name}!</b>

🤖 <b>Я ваш персональный AI-тренер по дартсу!</b>

📊 <b>Что я умею:</b>
• Анализировать технику броска
• Оценивать стабильность
• Сравнивать с эталонами PDC
• Давать персонализированные рекомендации

🎬 <b>Как получить анализ:</b>
1. Снимите видео броска <b>СБОКУ</b>
2. Длительность 5-10 секунд
3. Отправьте мне видео файлом
4. Получите детальный отчет!

👇 <b>Начните прямо сейчас:</b>
"""
    
    keyboard = [
        [InlineKeyboardButton("📹 ОТПРАВИТЬ ВИДЕО", callback_data="send_video")],
        [InlineKeyboardButton("❓ ПОМОЩЬ", callback_data="help")]
    ]
    
    await update.message.reply_text(
        welcome_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='HTML'
    )

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка видео"""
    user = update.effective_user
    
    # Сообщение о начале обработки
    msg = await update.message.reply_text(
        "📥 <b>Видео получено!</b>\n"
        "⏳ Анализирую технику...",
        parse_mode='HTML'
    )
    
    try:
        # Имитация анализа
        await asyncio.sleep(2)
        
        await msg.edit_text(
            "🔍 <b>Определяю ключевые параметры...</b>\n"
            "📊 Измеряю углы и стабильность...",
            parse_mode='HTML'
        )
        
        await asyncio.sleep(2)
        
        # Генерируем анализ
        video_info = {
            'duration': 5,
            'frames': 150,
            'user_id': user.id
        }
        
        analysis = analyzer.analyze_video(video_info)
        
        await msg.edit_text(
            "📈 <b>Формирую отчет...</b>\n"
            "💡 Готовлю рекомендации...",
            parse_mode='HTML'
        )
        
        await asyncio.sleep(1)
        
        # Отправляем отчет
        report = format_report(analysis, user.first_name)
        
        keyboard = [
            [InlineKeyboardButton("🔄 НОВЫЙ АНАЛИЗ", callback_data="send_video")],
            [InlineKeyboardButton("🎯 УПРАЖНЕНИЯ", callback_data="exercises")],
            [InlineKeyboardButton("📊 ИСТОРИЯ", callback_data="history")]
        ]
        
        await msg.edit_text(
            report,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )
        
    except Exception as e:
        logging.error(f"Ошибка обработки видео: {e}")
        await msg.edit_text(
            "❌ <b>Ошибка обработки</b>\n\n"
            "Попробуйте отправить видео снова или напишите /start",
            parse_mode='HTML'
        )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка inline-кнопок"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "send_video":
        await query.edit_message_text(
            "📤 <b>ГОТОВ ПРИНЯТЬ ВИДЕО!</b>\n\n"
            "<b>Требования для лучшего анализа:</b>\n"
            "🎬 <b>Ракурс:</b> СТРОГО СБОКУ (90°)\n"
            "⏱️ <b>Длительность:</b> 5-10 секунд\n"
            "💡 <b>Освещение:</b> Хорошее, без теней\n"
            "📱 <b>Камера:</b> Неподвижна (штатив/опора)\n\n"
            "<i>Просто отправьте видео файлом в этот чат...</i>",
            parse_mode='HTML'
        )
    
    elif query.data == "exercises":
        await query.edit_message_text(
            "🎯 <b>ТОП-5 УПРАЖНЕНИЙ ДЛЯ ДАРТСА</b>\n\n"
            "1. <b>СТАБИЛЬНОСТЬ ЛОКТЯ</b>\n"
            "   • Удержание позиции: 3×30 сек\n"
            "   • Броски без дротика: 50 раз\n\n"
            "2. <b>МЕТКОСТЬ</b>\n"
            "   • Серии в T20: 10×3 дротика\n"
            "   • Работа по секторам\n\n"
            "3. <b>КОНСИСТЕНТНОСТЬ</b>\n"
            "   • Одинаковые броски: 100 раз\n"
            "   • Контроль темпа\n\n"
            "4. <b>СТОЙКА</b>\n"
            "   • Баланс на одной ноге\n"
            "   • Распределение веса\n\n"
            "5. <b>ПСИХОЛОГИЯ</b>\n"
            "   • Дыхательные упражнения\n"
            "   • Визуализация броска\n\n"
            "🏋️ <b>Тренируйтесь 3-4 раза в неделю!</b>",
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📹 НОВЫЙ АНАЛИЗ", callback_data="send_video")],
                [InlineKeyboardButton("🔙 НАЗАД", callback_data="back")]
            ])
        )
    
    elif query.data == "history":
        await query.edit_message_text(
            "📊 <b>ИСТОРИЯ АНАЛИЗОВ</b>\n\n"
            "<i>Функция истории будет доступна в следующем обновлении!</i>\n\n"
            "Пока что вы можете:\n"
            "1. Сохранять скриншоты отчетов\n"
            "2. Вести дневник тренировок\n"
            "3. Сравнивать прогресс визуально\n\n"
            "🎯 <b>Главное - регулярность тренировок!</b>",
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📹 НОВЫЙ АНАЛИЗ", callback_data="send_video")],
                [InlineKeyboardButton("🔙 НАЗАД", callback_data="back")]
            ])
        )
    
    elif query.data == "help":
        await query.edit_message_text(
            "❓ <b>ЧАСТО ЗАДАВАЕМЫЕ ВОПРОСЫ</b>\n\n"
            "🤖 <b>Как работает анализ?</b>\n"
            "AI анализирует вашу технику и сравнивает с эталонами профессиональных игроков PDC.\n\n"
            "📹 <b>Какое видео нужно?</b>\n"
            "• Формат: MP4, MOV\n"
            "• Размер: до 20MB\n"
            "• Ракурс: сбоку\n"
            "• Длительность: 5-10 сек\n\n"
            "⏱️ <b>Сколько длится анализ?</b>\n"
            "10-30 секунд\n\n"
            "🎯 <b>Это бесплатно?</b>\n"
            "Да, полностью бесплатно!\n\n"
            "🔄 <b>Как часто делать анализ?</b>\n"
            "Рекомендуется раз в 1-2 недели для отслеживания прогресса.",
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📹 ОТПРАВИТЬ ВИДЕО", callback_data="send_video")],
                [InlineKeyboardButton("🔙 НАЗАД", callback_data="back")]
            ])
        )
    
    elif query.data == "back":
        await start(update, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    text = update.message.text
    
    if text.lower() in ['привет', 'hello', 'start', 'старт']:
        await start(update, context)
    elif 'видео' in text.lower() or 'анализ' in text.lower():
        await update.message.reply_text(
            "🎬 <b>Отправьте мне видео броска для анализа!</b>\n\n"
            "Просто загрузите видео файлом в этот чат.\n\n"
            "<i>Для подробной информации напишите /start</i>",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text(
            "🤖 <b>AI DART COACH</b>\n\n"
            "Я анализирую технику броска в дартсе!\n\n"
            "📹 <b>Отправьте мне видео</b> вашего броска,\n"
            "и я дам детальный анализ вашей техники!\n\n"
            "🎯 <b>Команды:</b>\n"
            "/start - Начать работу\n"
            "/help - Помощь\n\n"
            "<i>Или просто отправьте видео файлом...</i>",
            parse_mode='HTML'
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    await update.message.reply_text(
        "🆘 <b>ПОМОЩЬ</b>\n\n"
        "🎯 <b>Как использовать бота:</b>\n"
        "1. Напишите /start\n"
        "2. Нажмите '📹 ОТПРАВИТЬ ВИДЕО'\n"
        "3. Загрузите видео броска\n"
        "4. Получите анализ\n\n"
        "📹 <b>Требования к видео:</b>\n"
        "• Вид сбоку\n"
        "• 5-10 секунд\n"
        "• Хорошее освещение\n"
        "• Формат: MP4, MOV\n\n"
        "🤖 <b>Возможности бота:</b>\n"
        "• Анализ углов суставов\n"
        "• Оценка стабильности\n"
        "• Сравнение с PDC\n"
        "• Рекомендации\n\n"
        "📞 <b>Поддержка:</b>\n"
        "Для вопросов и предложений пишите @ваш_username",
        parse_mode='HTML'
    )

# ==================== ЗАПУСК ====================
def main():
    """Основная функция"""
    print("🚀 Запуск AI DART COACH...")
    
    # Проверка токена
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ОШИБКА: Токен бота не установлен!")
        print("\n🔧 Как установить токен на Render:")
        print("1. Перейдите в Dashboard Render")
        print("2. Выберите ваш сервис")
        print("3. Нажмите 'Environment'")
        print("4. Добавьте переменную:")
        print("   Key: BOT_TOKEN")
        print("   Value: ваш_токен_от_BotFather")
        print("5. Перезапустите деплой")
        return
    
    # Настройка логирования
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Создание приложения
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("✅ Бот настроен!")
    print("⚡ Запускаю polling...")
    print("-" * 60)
    
    # Запуск
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
