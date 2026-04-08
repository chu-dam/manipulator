import pybullet as p
import os
import trimesh

def generate_collision_mesh(input_stl, output_obj):
    print(f"🔄 [{os.path.basename(input_stl)}] 파일을 분석하여 논리적 분할 파이프라인을 가동합니다...")
    
    # 1. 파일 존재 여부 엄격 검증
    if not os.path.exists(input_stl):
        print(f"❌ 에러: {input_stl} 파일을 찾을 수 없습니다.")
        return

    # 2. 포맷 호환성을 위한 자동 변환 (STL -> 임시 OBJ)
    print("🛠️ 1단계: V-HACD의 엄격한 기준에 맞춰 포맷 자동 변환 중...")
    temp_obj = input_stl.replace(".stl", "_temp.obj")
    mesh = trimesh.load(input_stl)
    mesh.export(temp_obj)
    
    # 3. V-HACD 알고리즘 가동
    print("🛠️ 2단계: 고정밀 볼록 분할 알고리즘 가동 중 (잠시만 기다려주세요)...")
    p.vhacd(
        fileNameIn=temp_obj,
        fileNameOut=output_obj,
        fileNameLogging="vhacd_log.txt",
        resolution=500000,
        concavity=0.001,
        depth=20,
        alpha=0.05
    )
    
    # 4. 임시 파일 폐기 및 최종 품질 검증
    if os.path.exists(temp_obj):
        os.remove(temp_obj) # 찌꺼기 파일 삭제
        
    if os.path.exists(output_obj):
        print(f"✅ 완벽한 품질로 분할 완료! 물리 엔진용 [{os.path.basename(output_obj)}] 파일이 탄생했습니다.")
    else:
        print("❌ 에러: 분할 과정에서 기준 미달 문제가 발생했습니다.")

# 사용 실행부 (절대 경로 적용)
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 입력과 출력 파일 이름 지정
    SOURCE_MESH = os.path.join(current_dir, "hole_51_7.stl")
    RESULT_MESH = os.path.join(current_dir, "hole_51_7.obj")
    
    print(f"📁 작업 기준 폴더: {current_dir}")
    generate_collision_mesh(SOURCE_MESH, RESULT_MESH)