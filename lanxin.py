import requests
import uuid
import time
import json

# ========== 配置区域 ==========
AppKey = "YOUR_API_KEY==" 
DOMAIN = 'https://api-ai.vivo.com.cn'  
URI = '/api/v1/image_generation'
MODEL = "Doubao-Seedream-4.5"
MODULE = "aigc"
# ============================


def generate_image(prompt: str, size: str = None) -> dict:
    headers = {
        "Authorization": f"Bearer {AppKey}",
        "Content-Type": "application/json", 
    }
    url_params = {
        "module": MODULE,
        "request_id": str(uuid.uuid4()),         
        "system_time": int(time.time())    
    }
    
    body = {
        "model": MODEL,
        "prompt": prompt
    }
    if size:
        body["parameters"] = {"size": size}
    

    url = f"{DOMAIN}{URI}"
    try:
        response = requests.post(
            url,
            params=url_params,
            json=body,          
            headers=headers,
            timeout=30       
        )
        
        if response.status_code == 200:
            result = response.json()
            print("请求成功")
            return result
        else:
            print(f"请求失败: {response.status_code}")
            print(f"响应: {response.text}")
            return {"error": response.status_code, "detail": response.text}
            
    except requests.exceptions.Timeout:
        print("请求超时，请增大 timeout 或检查网络")
        return {"error": "timeout"}
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return {"error": "request_failed", "detail": str(e)}


if __name__ == '__main__':

    prompt = "我是CS选手Niko,生成一张我获得major冠军的照片，背景是一个大型电竞舞台，灯光璀璨，观众热情欢呼，我手持奖杯，表情兴奋，穿着战队制服，整体画面充满胜利的喜悦和激动人心的氛围"
    result = generate_image(prompt, size="2K")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("data", {}).get("image_url"):
        img_url = result["data"]["image_url"]
        print(f"生成成功！图片链接: {img_url}")
        # 可选：自动保存图片
        # img_data = requests.get(img_url).content
        # with open("output.jpg", "wb") as f:
        #     f.write(img_data)
        # print("💾 已保存为 output.jpg")